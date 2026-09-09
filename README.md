# factory-ai-manual

工場内PCで動く、シンプルな動画マニュアル作成ツールです。長い動画からのサイクル検出や、1サイクルごとの動画分割は行いません。

## 機能

- 動画をローカル保存
- OpenCVが映像変化から作業の開始・終了秒を自動検出
- 作業クリックで元動画の該当位置へジャンプ
- 動画下に作業別ガントチャートと現在位置を表示
- 選択中の作業だけを繰り返し再生
- Ollama + Phi-4による作業名・説明文の自動生成（必須）
- 自動生成後の手動修正
- SQLiteによる動画・作業・翻訳データの保存
- 完成後にArgos Translateで英語・ベトナム語・中国語へ翻訳

## 起動

```
1. 必要ソフトの確認
git --version
py -3.12 --version
ollama --version
認識されないものだけインストールします。
Gitがない場合
winget install --id Git.Git -e
Python 3.12がない場合
winget install --id Python.Python.3.12 -e
Ollamaがない場合
powershell -Command "irm https://ollama.com/install.ps1 | iex"
インストール後、コマンドプロンプトを一度閉じて、新しく開き直してください。
2. Phi-4をダウンロード
ollama pull phi4
確認：
ollama list
一覧に phi4 が表示されれば準備完了です。
3. 保存先を作成
mkdir "C:\factory-ai"
cd /d "C:\factory-ai"
4. GitHubから取得
git clone https://github.com/yusukebaba639579-cel/factory-ai-manual.git
プロジェクトへ移動：
cd factory-ai-manual
確認：
git status
5. Python仮想環境を作成
py -3.12 -m venv .venv
仮想環境を有効化：
.venv\Scripts\activate.bat
成功すると、行の先頭に (.venv) と表示されます。
6. ライブラリをインストール
python -m pip install --upgrade pip
pip install -r requirements.txt
MediaPipe、OpenCV、Flask、Argos Translateなどがインストールされます。数分かかる場合があります。
7. Ollamaの動作確認
ollama run phi4
入力待ちになったら、次を入力します。
こんにちは
回答が表示されたら終了します。
/bye
Ollamaアプリはバックグラウンドで動かしておいてください。
8. アプリを起動
プロジェクトフォルダにいることを確認：
cd /d "C:\factory-ai\factory-ai-manual"
仮想環境を有効化：
.venv\Scripts\activate.bat
起動：
python app.py
次の表示が出れば成功です。
Running on http://127.0.0.1:5051
ブラウザで開きます。
http://127.0.0.1:5051
9. 初回利用時
最初の動画解析時に、MediaPipeのモデルがダウンロードされます。
最初の翻訳時に、選択した言語のArgos Translateモデルがダウンロードされます。
そのため、初回だけインターネット接続が必要です。モデル取得後はローカルで動作します。
10. アプリの終了
アプリを起動したコマンドプロンプトで：
Ctrl + C
2回目以降の起動
git config --global --add safe.directory "%CD%"
git pull origin main
依存ライブラリも更新します。
cd C:\Users\y-baba.ADTEK-FUJI\factoryai\factory-ai-manual

call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python app.py

ブラウザ：
http://127.0.0.1:5051
```

ブラウザで `http://127.0.0.1:5051` を開きます。既存の `factory-ai`（5050番）とは別ポートです。

先に別のターミナルで `ollama run phi4` を実行してください。Ollama + Phi-4へ接続できない場合、マニュアルは生成されません。

`phi4` は映像を直接見るモデルではありません。作業区切りはOpenCVが映像変化から検出し、Phi-4は作業名・作業順・区間時間をもとに文章を生成します。

作業ラベルは「ねじ締め」「ねじの締付確認」「不適合確認」の3種類と「判定保留」です。作業数は固定せず、OpenCVの映像変化とMediaPipe Pose／Handsの動作変化から決定します。Phi-4の説明文は各作業1文・30文字以内です。

PDF・Word・Excel・CSV・TXTの手順書を動画と一緒に登録できます。手順書がある場合は、作業数・作業名・説明文を手順書から生成し、映像解析結果は各作業の時間位置を合わせるために使います。

正しい作業ラベルへ修正して「保存して学習」を押すと、動作特徴と正解ラベルがSQLiteへ保存されます。学習データが少ない間も手の検出量と動作量から3種類を暫定判定し、6件以上蓄積すると過去の近い動作を優先します。

翻訳はマニュアル完成後の閲覧画面で実行します。初回だけ選択言語のArgos翻訳モデルを取得し、以後はローカルで翻訳してSQLiteへ保存します。
