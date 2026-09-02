# API key
import os
from dotenv import load_dotenv
# from pathlib import Path



# 網頁基本設定
PAGE_TITLE = "文件摘要與對話機器人 | Nicole"
PAGE_ICON = "📄" 
TITLE = "📄 文件摘要與對話機器人"

# 頁尾 HTML 與 CSS（支援主題變色）
FOOTER_CSS = """
<style>
.custom-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    
    background-color: var(--background-color); 
    color: var(--text-color);
    
    text-align: center;
    padding: 8px 8px 8px 8px; /* 調整頁尾的內邊距: 上右下左 */
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
    <!-- <a href="https://linkedin.com" target="_blank">LinkedIn</a> -->
</div>
"""
