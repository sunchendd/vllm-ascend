import os
import sys
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Optional, List

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.spec_decode.suffix_decoding import \
    SuffixDecodingProposer as VllmSuffixDecodingProposer

from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType
from vllm.distributed import get_tensor_model_parallel_rank

# ============================================================================
# 环境变量配置
# ============================================================================
# VLLM_ASCEND_ADAPTIVE_SPEC: 是否启用自适应投机 (1=启用, 0=禁用, 默认0)
# VLLM_ASCEND_ADAPTIVE_THRESHOLD: 并发阈值，超过此值关闭投机
# VLLM_ASCEND_ADAPTIVE_LOG_INTERVAL: 日志打印间隔秒数 (默认30)
# VLLM_ASCEND_CALIBRATE_MAX_CONCURRENCY: 校准时的最大测试并发数 (默认64)

ADAPTIVE_SPEC_ENABLED = os.environ.get("VLLM_ASCEND_ADAPTIVE_SPEC", "0") == "1"
ADAPTIVE_LOG_INTERVAL = float(os.environ.get("VLLM_ASCEND_ADAPTIVE_LOG_INTERVAL", "30"))

# 阈值配置
# -1: 等待校准
# >=0: 具体阈值
_threshold_str = os.environ.get("VLLM_ASCEND_ADAPTIVE_THRESHOLD", "")
if _threshold_str:
    ADAPTIVE_THRESHOLD = int(_threshold_str)
    AUTO_CALIBRATE_ENABLED = False
else:
    # 默认阈值为-1，表示"等待校准完成"
    # 在校准完成前，处于保守策略(或者保持开启直到有结果)
    # 此处策略：在校准完成前，默认允许投机（假设低并发启动）
    ADAPTIVE_THRESHOLD = -1 
    AUTO_CALIBRATE_ENABLED = ADAPTIVE_SPEC_ENABLED

CALIBRATE_MAX_CONCURRENCY = int(os.environ.get("VLLM_ASCEND_CALIBRATE_MAX_CONCURRENCY", "64"))
CALIBRATE_PORT = int(os.environ.get("VLLM_ASCEND_CALIBRATE_PORT", "0"))
CALIBRATE_MODEL = os.environ.get("VLLM_ASCEND_CALIBRATE_MODEL", "")

CALIBRATE_CACHE_DIR = os.environ.get(
    "VLLM_ASCEND_CALIBRATE_CACHE_DIR",
    os.path.expanduser("~/.cache/vllm_ascend/calibration")
)
CALIBRATE_CACHE_ENABLED = os.environ.get("VLLM_ASCEND_CALIBRATE_CACHE_ENABLED", "1") == "1"


def _parse_port_from_cmdline() -> Optional[int]:
    """从命令行解析端口"""
    for i, arg in enumerate(sys.argv):
        if arg == '--port' and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
        if arg.startswith('--port='):
            try:
                return int(arg.split('=', 1)[1])
            except ValueError:
                pass
    return None


def _parse_model_from_cmdline() -> Optional[str]:
    """从命令行解析模型名"""
    for i, arg in enumerate(sys.argv):
        if arg == '--served-model-name' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith('--served-model-name='):
            return arg.split('=', 1)[1]
    return None


class CalibrationCache:
    """校准结果缓存管理器"""
    
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, model_name: str, tp_size: int, 
                       num_spec_tokens: int, max_concurrency: int) -> str:
        key_data = {
            "model": model_name,
            "tp_size": tp_size,
            "num_spec_tokens": num_spec_tokens,
            "max_concurrency": max_concurrency,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def get(self, model_name: str, tp_size: int, 
            num_spec_tokens: int, max_concurrency: int) -> Optional[dict]:
        if not self.enabled:
            return None
            
        cache_key = self._get_cache_key(model_name, tp_size, num_spec_tokens, max_concurrency)
        cache_file = self.cache_dir / f"calibration_{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"[CalibrationCache] Failed to load cache: {e}")
        return None
    
    def set(self, model_name: str, tp_size: int, num_spec_tokens: int,
            max_concurrency: int, threshold: int):
        if not self.enabled:
            return
            
        cache_key = self._get_cache_key(model_name, tp_size, num_spec_tokens, max_concurrency)
        cache_file = self.cache_dir / f"calibration_{cache_key}.json"
        
        data = {
            "threshold": threshold,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            # Metadata for verification
            "model": model_name,
            "config": {
                "tp": tp_size,
                "spec_tokens": num_spec_tokens,
                "max_concurrency": max_concurrency
            }
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"[CalibrationCache] Saved threshold={threshold} to {cache_file}")
        except Exception as e:
            logger.warning(f"[CalibrationCache] Failed to save cache: {e}")


class SuffixDecodingProposer(VllmSuffixDecodingProposer, Proposer):
    """
    Suffix Decoding Proposer with Adaptive Speculation
    
    Features:
    1. Dynamic Start/Stop based on concurrency threshold.
    2. Auto-calibration at startup (doubling probe + binary search).
    3. Caching of calibration results.
    """

    # Shared Class State
    _calibration_lock = threading.Lock()
    _calibration_done = False
    _calibrated_threshold: Optional[int] = None
    _calibration_in_progress = False
    
    # Mode control flags (shared across instances)
    _calibration_mode = False
    _force_skip_speculation = False
    
    _calibration_cache: Optional[CalibrationCache] = None

    def __init__(self, vllm_config, device, runner):
        super().__init__(vllm_config)
        self.name = SpecDcodeType.SUFFIX
        self.device = device
        self.runner = runner
        
        self._adaptive_enabled = ADAPTIVE_SPEC_ENABLED
        self._adaptive_threshold = ADAPTIVE_THRESHOLD
        self._last_log_time = 0.0
        self._log_interval = ADAPTIVE_LOG_INTERVAL
        
        # Init Cache
        if SuffixDecodingProposer._calibration_cache is None:
            SuffixDecodingProposer._calibration_cache = CalibrationCache(
                CALIBRATE_CACHE_DIR, CALIBRATE_CACHE_ENABLED
            )
            
        # 1. Config Detection
        self._detect_config(vllm_config)
        
        # 2. Check Cache / Init Calibration State
        self._need_first_run_calibration = False
        
        if self._adaptive_enabled and AUTO_CALIBRATE_ENABLED and not SuffixDecodingProposer._calibration_done:
            # Check cache first
            cached = SuffixDecodingProposer._calibration_cache.get(
                self._calibrate_model,
                self._tp_size,
                self._num_spec_tokens,
                self._calibrate_max_concurrency
            )
            
            if cached and 'threshold' in cached:
                th = cached['threshold']
                with SuffixDecodingProposer._calibration_lock:
                    SuffixDecodingProposer._calibrated_threshold = th
                    SuffixDecodingProposer._calibration_done = True
                self._adaptive_threshold = th
                logger.info(f"[AdaptiveSpec] Loaded cached threshold: {th}")
            else:
                self._need_first_run_calibration = True
                logger.info(f"[AdaptiveSpec] No cache found. Scheduled calibration.")

        # Stats
        self._spec_enabled_count = 0
        self._spec_disabled_count = 0

    def _detect_config(self, vllm_config):
        # Port
        self._calibrate_port = CALIBRATE_PORT if CALIBRATE_PORT > 0 else (_parse_port_from_cmdline() or 8000)
        
        # Model
        if CALIBRATE_MODEL:
            self._calibrate_model = CALIBRATE_MODEL
        else:
            self._calibrate_model = _parse_model_from_cmdline() or "default"
            # Try to get from config if possible
            served = getattr(vllm_config.model_config, 'served_model_name', None) if hasattr(vllm_config, 'model_config') else None
            if served:
                self._calibrate_model = served[0] if isinstance(served, list) else served

        # TP & Spec Config
        self._tp_size = getattr(vllm_config.parallel_config, 'tensor_parallel_size', 1)
        self._num_spec_tokens = getattr(vllm_config.speculative_config, 'num_speculative_tokens', 15) if vllm_config.speculative_config else 15
        self._calibrate_max_concurrency = CALIBRATE_MAX_CONCURRENCY

    def load_model(self, *args, **kwargs):
        pass

    @torch.inference_mode()
    def dummy_run(self, *args, **kwargs):
        pass

    # --- Mode Control ---

    def enable_speculation(self):
        SuffixDecodingProposer._force_skip_speculation = False
        
    def disable_speculation(self):
        SuffixDecodingProposer._force_skip_speculation = True

    def _should_skip_speculation(self, num_reqs: int) -> bool:
        """Core logic for Adaptive Speculation"""
        # 1. Calibration Mode: controlled by Calibrator strictly
        if SuffixDecodingProposer._calibration_mode:
            return SuffixDecodingProposer._force_skip_speculation
            
        # 2. Global Force Skip (safety switch)
        if SuffixDecodingProposer._force_skip_speculation:
            return True
            
        # 3. If Adaptive Disabled -> Always Speculate (unless configured otherwise in upper layers)
        if not self._adaptive_enabled:
            return False
            
        # 4. If waiting for calibration (-1) -> Default to Speculate (allow warmup)
        if self._adaptive_threshold == -1:
            return False
            
        # 5. Threshold Logic
        return num_reqs > self._adaptive_threshold

    # --- Calibration ---

    def _run_calibration_async(self):
        """Invoke the SmartWarmupCalibrator in a background thread"""
        def _job():
            try:
                from vllm_ascend.spec_decode.warmup_calibrator import SmartWarmupCalibrator
                
                logger.info(f"[AdaptiveSpec] Starting calibration on port {self._calibrate_port}...")
                
                # Enter exclusive mode
                SuffixDecodingProposer._calibration_mode = True
                
                calibrator = SmartWarmupCalibrator(
                    base_url=f"http://127.0.0.1:{self._calibrate_port}",
                    model_name=self._calibrate_model,
                    max_concurrency=self._calibrate_max_concurrency
                )
                
                calibrator.set_mode_callbacks(self.enable_speculation, self.disable_speculation)
                
                result = calibrator.run_calibration()
                
                threshold = -1
                if result:
                    threshold = result.optimal_threshold
                    logger.info(f"[AdaptiveSpec] Calibration Success! Optimal threshold: {threshold}")
                    
                    # Update cache
                    SuffixDecodingProposer._calibration_cache.set(
                        self._calibrate_model, self._tp_size, 
                        self._num_spec_tokens, self._calibrate_max_concurrency,
                        threshold
                    )
                else:
                    logger.error("[AdaptiveSpec] Calibration Failed. Defaulting to keep Speculation ON.")
                    threshold = 9999 # Safe fallback
                
                # Update Global State
                with SuffixDecodingProposer._calibration_lock:
                    SuffixDecodingProposer._calibrated_threshold = threshold
                    SuffixDecodingProposer._calibration_done = True
                    # Also update ENV for other processes if needed
                    os.environ["VLLM_ASCEND_ADAPTIVE_THRESHOLD"] = str(threshold)

            except Exception as e:
                logger.error(f"[AdaptiveSpec] Calibration crashed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                SuffixDecodingProposer._calibration_mode = False
                SuffixDecodingProposer._force_skip_speculation = False # Reset
                SuffixDecodingProposer._calibration_in_progress = False

        t = threading.Thread(target=_job, daemon=True)
        t.start()

    def generate_token_ids(self,
                           valid_sampled_token_ids,
                           sampling_metadata=None,
                           scheduler_output=None,
                           spec_decode_metadata=None,
                           positions=None,
                           num_scheduled_tokens=None,
                           hidden_states=None,
                           aux_hidden_states=None) -> list[list[int]]:
        
        # --- Trigger Calibration Once ---
        if self._need_first_run_calibration:
            self._need_first_run_calibration = False
            
            tp_rank = get_tensor_model_parallel_rank()
            should_run = False
            
            with SuffixDecodingProposer._calibration_lock:
                if SuffixDecodingProposer._calibration_done:
                     pass # Already done by someone else
                elif tp_rank == 0 and not SuffixDecodingProposer._calibration_in_progress:
                     SuffixDecodingProposer._calibration_in_progress = True
                     should_run = True
            
            if should_run:
                logger.info("[AdaptiveSpec] Triggering initial calibration...")
                self._run_calibration_async()
                
        # --- Update Local Threshold from Global ---
        with SuffixDecodingProposer._calibration_lock:
             if SuffixDecodingProposer._calibrated_threshold is not None:
                 if self._adaptive_threshold != SuffixDecodingProposer._calibrated_threshold:
                     self._adaptive_threshold = SuffixDecodingProposer._calibrated_threshold
                     logger.info(f"[AdaptiveSpec] Local threshold updated to {self._adaptive_threshold}")

        # --- Runtime Logic ---
        input_batch = self.runner.input_batch
        num_reqs = input_batch.num_reqs
        
        skip = self._should_skip_speculation(num_reqs)
        
        # Logging
        if skip:
            self._spec_disabled_count += 1
        else:
            self._spec_enabled_count += 1
            
        cur_time = time.time()
        if cur_time - self._last_log_time > self._log_interval:
            self._last_log_time = cur_time
            total = self._spec_disabled_count + self._spec_enabled_count
            ratio = (self._spec_enabled_count / total * 100) if total > 0 else 0
            logger.info(f"[AdaptiveSpec] Status: P={num_reqs}, Th={self._adaptive_threshold}, Skip={skip}, SpecRate={ratio:.1f}%")

        if skip:
            # Return empty lists to signal "no speculation"
            return [[] for _ in range(len(valid_sampled_token_ids))]
            
        return self.propose(input_batch, valid_sampled_token_ids)
