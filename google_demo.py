from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build

# 配置
KEY_PATH = Path.home() / ".credentials" / "credentials.json"
FILE_ID = "1ZawDnCnk8q4lUVrDhxPgwcK-aW_una4nSV4_imiWsYA"

# PDF格式对应的MIME类型
MIME_TYPE = "application/pdf"

try:
    if not KEY_PATH.exists():
        raise FileNotFoundError(f"未找到密钥文件: {KEY_PATH}")

    credentials = service_account.Credentials.from_service_account_file(
        str(KEY_PATH),
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    drive_service = build("drive", "v3", credentials=credentials)

    # 1. 先获取文件的原始名称
    file_metadata = drive_service.files().get(fileId=FILE_ID).execute()
    original_filename = file_metadata.get("name", "unknown_file")  # 获取原文件名
    print(f"获取到原文件名称：{original_filename}")

    # 2. 构建保存路径（原文件名 + .pdf 扩展名）
    save_path = f"{original_filename}.pdf"

    # 3. 导出为PDF
    request = drive_service.files().export_media(
        fileId=FILE_ID,
        mimeType=MIME_TYPE
    )

    # 4. 用原文件名保存到本地
    with open(save_path, "wb") as f:
        response = request.execute()
        f.write(response)

    print(f"文件已成功导出为PDF：{save_path}")
    print(f"保存路径：{Path(save_path).resolve()}")

except Exception as e:
    print("运行时出错：", str(e))
