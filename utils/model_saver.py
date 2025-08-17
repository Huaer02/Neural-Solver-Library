import os
import torch
import logging
from typing import Dict, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelSaver:
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_name: str = "best_model",
        monitor_metric: str = "test_l2",
        mode: str = "min",
        patience: int = 10,
        verbose: bool = True
    ):
        """
        Args:
            save_dir: 保存目录
            save_name: 保存文件名前缀
            monitor_metric: 监控的指标名称 (如 'test_l2', 'val_loss', 'mae' 等)
            mode: 'min' 表示越小越好, 'max' 表示越大越好
            patience: 多少个epoch没有改善后停止更新最佳模型
            verbose: 是否打印信息
        """
        self.save_dir = save_dir
        self.save_name = save_name
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        
        os.makedirs(save_dir, exist_ok=True)
        
        if mode == "min":
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best
        elif mode == "max":
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'min' or 'max'.")
        
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.best_model_path = None
    
    def update(self, model: torch.nn.Module, metrics: Dict[str, float], epoch: int) -> bool:
        """
        检查并更新最佳模型
        
        Args:
            model: PyTorch模型
            metrics: 指标字典，例如 {'val_loss': 0.1, 'test_l2': 0.05, 'mae': 0.02}
            epoch: 当前epoch
            
        Returns:
            bool: 是否保存了新的最佳模型
        """
        if self.monitor_metric not in metrics:
            if self.verbose:
                logger.warning(f"监控指标 '{self.monitor_metric}' 不在metrics中: {list(metrics.keys())}")
            return False
        
        current_score = metrics[self.monitor_metric]
        
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_name}_epoch{epoch}_{timestamp}.pt"
            self.best_model_path = os.path.join(self.save_dir, filename)
            
            torch.save(model.state_dict(), self.best_model_path)
            
            if self.verbose:
                logger.info(f"保存新的最佳模型: {filename} ({self.monitor_metric}: {current_score:.6f})")
                
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def save_final(self, model: torch.nn.Module, epoch: int):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_name}_final_epoch{epoch}_{timestamp}.pt"
        filepath = os.path.join(self.save_dir, filename)
        
        torch.save(model.state_dict(), filepath)
        
        if self.verbose:
            logger.info(f"保存最终模型: {filename}")
    
    def load_best_model(self, model: torch.nn.Module, map_location: str = "cpu"):
        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            raise FileNotFoundError("没有找到最佳模型文件")
        
        model.load_state_dict(torch.load(self.best_model_path, map_location=map_location))
        
        if self.verbose:
            logger.info(f"加载最佳模型: {os.path.basename(self.best_model_path)}")
    
    def get_best_model_path(self) -> Optional[str]:
        return self.best_model_path
    
    def should_stop_early(self) -> bool:
        return self.epochs_without_improvement >= self.patience
    
    def get_summary(self) -> Dict:
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
            'best_model_path': self.best_model_path,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode
        }
