"""
启动网格搜索UI界面的脚本
"""

import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入UI模块
from scripts.grid_search_ui import GridSearchUI
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格，更接近Windows 11
    ui = GridSearchUI()
    sys.exit(app.exec()) 