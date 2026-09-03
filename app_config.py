# API key
import os
from dotenv import load_dotenv
# from pathlib import Path



# 網頁基本設定
PAGE_TITLE = "文件摘要與對話機器人 | Nicole"
PAGE_ICON = "📄" 
TITLE = "📄 文件摘要與對話機器人"


# Sidebar
SIDEBAR_CSS = """
    <style>
    /* 設定側邊欄的最小與最大寬度 */
    [data-testid="stSidebar"] {
        min-width: 40%;
        max-width: 500px;

        z-index: 990;   

    }
    </style>
    """

# 頁尾 HTML 與 CSS（支援主題變色）
FOOTER_CSS = """
<style>
.custom-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    
    background-color: var(--secondary-background-color, #1e293b) !important;
    color: var(--text-color);
    
    text-align: center;
    padding: 4px 4px 4px 4px; /* 調整頁尾的內邊距: 上右下左 */
    font-size: 14px;
    z-index: 999;   /* 讓頁尾置於最上層 */
    
    border-top: 1px solid rgba(190, 190, 190, 0.5); 
}
.custom-footer a {
    /* link color */
    color: var(--primary-color); 
    text-decoration: none;
}
</style>
<div class="custom-footer">
    Built with Streamlit ‧ Created by <b>Nicole</b> ‧ nicoletuatie@gmail.com
</div>
"""
