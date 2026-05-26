"""
setup_kaggle.py
===============
首次使用前請執行此腳本，完成 Kaggle API 憑證設定與資料集驗證。

執行方式：
  python setup_kaggle.py

支援兩種環境：
  - 本機 (Windows / macOS / Linux)
  - Google Colab
"""

import os
import sys
import subprocess
import platform


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────

def print_banner():
    print("=" * 55)
    print("   Kaggle API 設定工具  |  OULAD 資料集")
    print("=" * 55)


def detect_env():
    """偵測執行環境（用 importlib 避免 Pylance 的 module-not-found 警告）"""
    import importlib.util
    if importlib.util.find_spec("google.colab") is not None:
        return "colab"
    return "local"


def install_package(pkg):
    """靜默安裝套件"""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def prompt_token():
    """引導用戶輸入 Kaggle API Token"""
    print("\n📋 步驟 1：取得 Kaggle API Token")
    print("   1. 前往 https://www.kaggle.com")
    print("   2. 點右上角頭像 → Settings")
    print("   3. 找到「API」區塊 → 點「Create New Token」")
    print("   4. 複製畫面上出現的 Token 字串（格式：KGAT_xxxxxxxx）")
    print()

    while True:
        token = input("請貼上你的 Kaggle Token：").strip()
        if not token:
            print("❌ Token 不能為空，請重新輸入。")
            continue
        if len(token) < 10:
            print("❌ Token 格式看起來不對，請確認後重新輸入。")
            continue
        return token


def setup_local(token):
    """本機環境：寫入系統環境變數（永久生效）"""
    print("\n⚙️  步驟 2：設定本機環境變數")

    # 同時設定給當前 process（本次執行立即生效）
    os.environ["KAGGLE_TOKEN"] = token

    system = platform.system()

    if system == "Windows":
        # 寫入 Windows 使用者層級環境變數（重啟 shell 後永久生效）
        subprocess.run(
            ["powershell", "-Command",
             f'[Environment]::SetEnvironmentVariable("KAGGLE_TOKEN", "{token}", "User")'],
            check=True,
            capture_output=True,
        )
        print("   ✅ Windows 使用者環境變數已設定（重新開啟 terminal 後永久生效）")

    elif system in ("Linux", "Darwin"):
        # 偵測使用哪個 shell rc 檔
        shell = os.environ.get("SHELL", "")
        if "zsh" in shell:
            rc_file = os.path.expanduser("~/.zshrc")
        else:
            rc_file = os.path.expanduser("~/.bashrc")

        export_line = f'\nexport KAGGLE_TOKEN="{token}"  # Added by setup_kaggle.py\n'

        # 避免重複寫入
        try:
            with open(rc_file, "r") as f:
                existing = f.read()
        except FileNotFoundError:
            existing = ""

        if "KAGGLE_TOKEN" not in existing:
            with open(rc_file, "a") as f:
                f.write(export_line)
            print(f"   ✅ 已寫入 {rc_file}（重新開啟 terminal 後永久生效）")
        else:
            print(f"   ℹ️  {rc_file} 中已有 KAGGLE_TOKEN，跳過寫入。")
    else:
        print(f"   ℹ️  未知系統 {system}，只設定當前 process 的環境變數。")


def setup_colab(token):
    """Colab 環境：設定當前 session 的環境變數"""
    print("\n⚙️  步驟 2：設定 Colab Session 環境變數")
    os.environ["KAGGLE_TOKEN"] = token
    print("   ✅ 已設定（僅限本次 Colab Session，斷線後需重新執行）")


def verify_download():
    """下載少量資料以驗證設定是否成功"""
    print("\n🔍 步驟 3：驗證 Kaggle 連線與下載")

    try:
        install_package("kagglehub")

        # 用 importlib 動態載入，避免 Pylance 的 module-not-found 警告
        import importlib
        kagglehub = importlib.import_module("kagglehub")

        print("   📥 嘗試下載 OULAD 資料集（首次需等待，之後有快取）...")
        dataset_path = kagglehub.dataset_download(
            "anlgrbz/student-demographics-online-education-dataoulad"
        )
        print(f"   ✅ 下載成功！資料集路徑：{dataset_path}")

        # 找到並預覽 studentInfo.csv
        target_file = None
        for root, dirs, files in os.walk(dataset_path):
            for fname in files:
                if fname == "studentInfo.csv":
                    target_file = os.path.join(root, fname)
                    break

        if target_file:
            install_package("pandas")
            import pandas as pd
            df = pd.read_csv(target_file, nrows=3)
            print("\n   📄 studentInfo.csv 前 3 筆資料預覽：")
            print(df.to_string(index=False))
            print(f"\n   ✅ 驗證完成！共有欄位：{list(df.columns)}")
        else:
            print("   ⚠️  找不到 studentInfo.csv，但資料集已下載成功。")

    except Exception as e:
        print(f"\n   ❌ 驗證失敗：{e}")
        print("   請確認 Token 是否正確，或是否已在 Kaggle 接受資料集使用條款。")
        sys.exit(1)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    print_banner()

    env = detect_env()
    print(f"\n🖥️  偵測到執行環境：{'Google Colab' if env == 'colab' else '本機'}")

    # 步驟 1：取得 Token
    token = prompt_token()

    # 步驟 2：依環境設定憑證
    if env == "colab":
        setup_colab(token)
    else:
        setup_local(token)

    # 步驟 3：驗證下載
    verify_download()

    print("\n" + "=" * 55)
    print("🎉 設定完成！現在可以執行 LSTM-FFF.ipynb 了。")
    print("=" * 55)


if __name__ == "__main__":
    main()
