"""
vLLM-Ascend 投机推理自动校准预热模块

本模块负责在服务启动时自动进行性能校准，确定投机推理的最佳并发阈值。
采用"倍增探测 + 二分搜索"的策略，快速且准确地找到性能交叉点。
"""

import os
import time
import statistics
import traceback
import requests
from typing import Optional, Tuple, Dict, List, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event, Lock

from vllm.logger import logger

# 复用Session以提高HTTP请求性能
RequestsSession = requests.Session()

@dataclass
class CalibrationMetric:
    """单个测试点的性能指标"""
    tps: float          # 吞吐量 (tokens/sec)
    latency: float      # 平均延迟 (ms)
    cv: float           # 变异系数 (稳定性指标)
    samples: int        # 样本数
    error_rate: float   # 错误率

@dataclass
class ComparisonResult:
    """Spec vs Direct 对比结果"""
    concurrency: int
    spec_metric: CalibrationMetric
    direct_metric: CalibrationMetric
    winner: str         # 'spec', 'direct', 'tie'
    improvement: float  # spec relative to direct (e.g., 0.1 means 10% faster)

@dataclass
class WarmupCalibrationResult:
    """最终校准结果"""
    optimal_threshold: int
    comparisons: Dict[int, ComparisonResult]
    confidence: float
    message: str

class SmartWarmupCalibrator:
    """
    智能预热校准器
    
    核心逻辑：
    1. 预热：确保服务JIT编译完成。
    2. 探测：对比 Speculative Decoding 和 Direct Decoding 的性能。
    3. 搜索：找到 Spec 性能优于 Direct 的最大并发数 (Threshold)。
       - 策略：先进行倍增探测 (1, 2, 4, 8, 16...) 确定区间。
       - 然后在区间内进行二分搜索精确打击。
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        model_name: str = "default",
        max_concurrency: int = 64,
        input_length: int = 256,
        output_length: int = 128,
        timeout: float = 300.0,
        # 统计配置
        min_samples: int = 5,
        max_samples: int = 20,
        cv_threshold: float = 0.15,    # 变异系数阈值
        duration_per_test: int = 5,    # 单点测试持续时间(秒)
        winner_margin: float = 0.03,   # 优胜判定阈值 (3%)
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.input_length = input_length
        self.output_length = output_length
        self.timeout = timeout
        
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.cv_threshold = cv_threshold
        self.duration_per_test = duration_per_test
        self.winner_margin = winner_margin
        
        # 回调函数
        self._enable_spec_callback: Optional[Callable] = None
        self._disable_spec_callback: Optional[Callable] = None
        self._current_spec_mode = True 
        
        # 缓存测量结果避免重复测试
        self._cache: Dict[str, CalibrationMetric] = {} # key: f"{concurrency}_{mode}"

    def set_mode_callbacks(self, enable_spec: Callable, disable_spec: Callable):
        self._enable_spec_callback = enable_spec
        self._disable_spec_callback = disable_spec

    def _generate_prompt(self) -> str:
        """生成具有代表性的中文Prompt"""
        base = "人工智能技术的飞速发展正在深刻改变人类社会的方方面面，从自动驾驶到智能医疗，深度学习算法的应用无处不在。"
        prompt = base
        # 填充到目标长度
        while len(prompt) < self.input_length * 2:
            prompt += base
        return prompt[:self.input_length]

    def _send_request(self, prompt: str) -> Tuple[bool, float, int]:
        """发送请求并计算延迟"""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.output_length,
            "temperature": 0.0, # 确定性
            "ignore_eos": True,
        }
        
        try:
            start = time.perf_counter()
            resp = RequestsSession.post(url, json=payload, timeout=self.timeout)
            latency = (time.perf_counter() - start) * 1000 # ms
            
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                tokens = usage.get("completion_tokens", self.output_length)
                return True, latency, tokens
            else:
                logger.warning(f"[Calibrator] Request failed: {resp.status_code}")
                return False, 0, 0
        except Exception as e:
            return False, 0, 0

    def _warmup_service(self):
        """
        全量预热
        发送一批请求，确保 max_concurrency 下的计算图都被触发和编译。
        """
        logger.info(f"[Calibrator] 正在进行服务预热 (Max Concurrency={self.max_concurrency})...")
        prompt = self._generate_prompt()
        
        concurrencies_to_warmup = [1, 4]
        if self.max_concurrency > 4:
            concurrencies_to_warmup.append(self.max_concurrency)
            
        for c in concurrencies_to_warmup:
            # 切换两种模式各跑一次
            for mode in [True, False]:
                self._switch_mode(mode)
                with ThreadPoolExecutor(max_workers=c) as executor:
                    list(executor.map(lambda _: self._send_request(prompt), range(c)))
                    
        logger.info("[Calibrator] 服务预热完成")

    def _switch_mode(self, enable_spec: bool):
        """切换 speculative/direct 模式"""
        if enable_spec == self._current_spec_mode:
            return

        if enable_spec:
            if self._enable_spec_callback:
                self._enable_spec_callback()
        else:
            if self._disable_spec_callback:
                self._disable_spec_callback()
        
        self._current_spec_mode = enable_spec
        # 模式切换后等待一小段时间让backend状态同步
        time.sleep(0.5)

    def _measure_point(self, concurrency: int) -> CalibrationMetric:
        """测量特定并发下的性能"""
        mode_str = "spec" if self._current_spec_mode else "direct"
        cache_key = f"{concurrency}_{mode_str}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._generate_prompt()
        samples_tps = []
        samples_lat = []
        errors = 0
        total_reqs = 0
        
        # 预热当前并发级别 (1 round)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            list(executor.map(lambda _: self._send_request(prompt), range(concurrency)))

        start_measure_time = time.perf_counter()
        
        while True:
            # 每轮并发请求
            batch_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(self._send_request, prompt) for _ in range(concurrency)]
                results = [f.result() for f in as_completed(futures)]
            
            batch_duration = time.perf_counter() - batch_start
            
            # 统计本轮
            batch_tokens = 0
            for success, lat, tokens in results:
                total_reqs += 1
                if success:
                    batch_tokens += tokens
                    samples_lat.append(lat)
                else:
                    errors += 1
            
            if batch_duration > 0:
                # 本轮 TPS
                current_tps = batch_tokens / batch_duration
                samples_tps.append(current_tps)
            
            # 检查退出条件
            elapsed = time.perf_counter() - start_measure_time
            if elapsed >= self.duration_per_test and len(samples_tps) >= self.min_samples:
                # 计算CV
                mean = statistics.mean(samples_tps)
                if mean > 0:
                    cv = statistics.stdev(samples_tps) / mean if len(samples_tps) > 1 else 0.0
                    if cv <= self.cv_threshold:
                        break
            
            if elapsed > self.duration_per_test * 2: # 超时强制退出
                break
                
        # 汇总结果
        if not samples_tps:
            return CalibrationMetric(0, 0, 0, 0, 1.0)
            
        mean_tps = statistics.mean(samples_tps)
        mean_lat = statistics.mean(samples_lat) if samples_lat else 0
        cv = statistics.stdev(samples_tps) / mean_tps if len(samples_tps) > 1 and mean_tps > 0 else 0.0
        error_rate = errors / total_reqs if total_reqs > 0 else 0
        
        metric = CalibrationMetric(mean_tps, mean_lat, cv, len(samples_tps), error_rate)
        self._cache[cache_key] = metric
        
        logger.debug(f"[Calibrator] P={concurrency} Mode={mode_str.upper()} -> TPS={mean_tps:.1f}, Lat={mean_lat:.1f}ms")
        return metric

    def _compare(self, concurrency: int) -> ComparisonResult:
        """对比两种模式"""
        # 测 Direct
        self._switch_mode(False)
        m_direct = self._measure_point(concurrency)
        
        # 测 Spec
        self._switch_mode(True)
        m_spec = self._measure_point(concurrency)
        
        # 判定
        winner = "tie"
        improvement = 0.0
        
        if m_direct.tps > 0:
            improvement = (m_spec.tps - m_direct.tps) / m_direct.tps
            
        if improvement > self.winner_margin:
            winner = "spec"
        elif improvement < -self.winner_margin:
            winner = "direct"
            
        logger.info(f"[Calibrator] 对比 P={concurrency}: "
                   f"Spec={m_spec.tps:.1f} vs Direct={m_direct.tps:.1f} "
                   f"(Diff: {improvement:+.1%}) -> Winner: {winner.upper()}")
                   
        return ComparisonResult(concurrency, m_spec, m_direct, winner, improvement)

    def run_calibration(self) -> Optional[WarmupCalibrationResult]:
        """运行完整校准流程"""
        if not self._wait_for_service():
            return None
            
        logger.info("="*60)
        logger.info("启动智能性能校准 (Smart Calibration)")
        logger.info(f"配置: MaxConcurrency={self.max_concurrency}, CV={self.cv_threshold}")
        logger.info("="*60)
        
        try:
            self._warmup_service()
            self._cache.clear()
            comparisons = {}
            
            # 策略：倍增探测
            # 1, 2, 4, 8, 16, 32... 
            # 找到第一个 Spec 变慢的点，确定交叉区间
            
            # 第一点：必测 P=1
            r1 = self._compare(1)
            comparisons[1] = r1
            
            # 如果 P=1 时 Direct 就更好，那全程 Direct
            if r1.winner == "direct":
                logger.info("[Calibrator] 低并发(P=1)下 Direct 更优，无需继续测试。")
                return WarmupCalibrationResult(0, comparisons, 1.0, "Direct mode preferred at P=1")
                
            # 倍增查找 Spec 转折点
            curr = 1
            interval_start = 1
            interval_end = self.max_concurrency
            found_crossover_interval = False
            
            while curr * 2 <= self.max_concurrency:
                next_p = curr * 2
                res = self._compare(next_p)
                comparisons[next_p] = res
                
                if res.winner == "direct":
                    # 发现Spec变慢了，交叉点在 [curr, next_p] 之间
                    interval_start = curr
                    interval_end = next_p
                    found_crossover_interval = True
                    break
                elif res.winner == "tie":
                    # 并列，可能即将交叉
                    interval_start = curr
                    # 继续往后看
                
                curr = next_p
                
            # 如果倍增跑完了都没发现 Direct 赢，再测一下 Max Concurrency
            if not found_crossover_interval:
                if self.max_concurrency not in comparisons:
                    r_max = self._compare(self.max_concurrency)
                    comparisons[self.max_concurrency] = r_max
                    if r_max.winner == "direct":
                        interval_start = curr
                        interval_end = self.max_concurrency
                        found_crossover_interval = True
                    elif r_max.winner == "spec":
                         # Max 也是 Spec 赢 -> 全程 Spec
                        return WarmupCalibrationResult(999999, comparisons, 1.0, "Spec mode preferred up to max concurrency")
            
            # 二分搜索精确点
            # 在 [interval_start, interval_end] 之间找 Spec 最后一次赢/平的点
            logger.info(f"[Calibrator] 锁定交叉区间: [{interval_start}, {interval_end}]，开始二分搜索...")
            
            best_threshold = interval_start
            low = interval_start + 1
            high = interval_end - 1 # 边界已经被测试过了
            
            while low <= high:
                mid = (low + high) // 2
                if mid in comparisons:
                    res = comparisons[mid]
                else:
                    res = self._compare(mid)
                    comparisons[mid] = res
                
                if res.winner in ["spec", "tie"]:
                    best_threshold = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            # 最终确认
            logger.info("="*60)
            logger.info(f"校准完成. 最佳阈值 Threshold = {best_threshold}")
            logger.info(f"含义: 并发数 <= {best_threshold} 时开启投机推理")
            logger.info("="*60)
            
            return WarmupCalibrationResult(
                optimal_threshold=best_threshold,
                comparisons=comparisons,
                confidence=0.9,
                message=f"Crossover found at P={best_threshold}"
            )
            
        except Exception as e:
            logger.error(f"[Calibrator] 校准过程出错: {e}")
            traceback.print_exc()
            return None
        finally:
            self._switch_mode(False) # 默认切回 Direct 安全模式，由上层根据阈值决定

    def _wait_for_service(self) -> bool:
        start = time.time()
        while time.time() - start < 120:
            try:
                if requests.get(f"{self.base_url}/health", timeout=1).status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        logger.error("[Calibrator] 等待服务健康检查超时")
        return False

# -----------------------------------------------------------------------------
# 简化调用接口
# -----------------------------------------------------------------------------
def run_warmup_calibration(
    port: int,
    model_name: str,
    max_concurrency: int,
    enable_callback: Callable,
    disable_callback: Callable
) -> int:
    """可以直接调用的帮助函数"""
    calibrator = SmartWarmupCalibrator(
        base_url=f"http://127.0.0.1:{port}",
        model_name=model_name,
        max_concurrency=max_concurrency
    )
    calibrator.set_mode_callbacks(enable_callback, disable_callback)
    res = calibrator.run_calibration()
    if res:
        return res.optimal_threshold
    return -1 # Failed
