# Long's part: Facial Recognition with Emotion and Liveness

Code for Long's assigned contribution: classification-based face embeddings, facial expression
classification, registration, attendance UI, and registration image-quality checks.

**Chưa có model đã train hoặc kết quả thực nghiệm trong gói này.** Cần dữ liệu ảnh thật để
chạy hai notebook. Metric learning và anti-spoofing do thành viên còn lại triển khai.

## 1. Bắt đầu trên Google Colab

1. Giải nén gói source trên máy. Giữ lại file `Long_Facial_Attendance.zip` để đưa lên Drive.
2. Trong My Drive, tạo thư mục `Long_Face_Project`.
3. Đưa **file ZIP source còn nguyên** vào thư mục này, tên `Long_Facial_Attendance.zip`.
4. Đưa ZIP dữ liệu face vào cùng thư mục, đặt tên `face_dataset.zip`, hoặc sửa đường dẫn trong notebook.
5. Mở Google Colab, chọn **File > Upload notebook**, mở `notebooks/01_classification.ipynb`.
6. Chọn GPU runtime. Chạy các cell từ trên xuống. Notebook tự giải nén source và dữ liệu vào Colab.
7. Kiểm tra ảnh minh họa, số ID và số ảnh. Nếu cấu trúc không khớp, dừng để sửa đường dẫn.
8. Chạy train, đánh giá classification, rồi đánh giá verification.
9. Kết quả được lưu trong Drive, tại `Long_Face_Project/runs/face_classification`.
10. Chạy cell export cuối cùng. Đổi `DOWNLOAD_RESULTS = True` để tải ZIP kết quả về máy.
11. Tải notebook đã chạy bằng **File > Download > Download .ipynb**, thay vào đúng file trong repo.

Sau đó làm tương tự với `notebooks/03_emotion.ipynb`. Notebook emotion cần ZIP dữ liệu riêng,
tên mặc định `emotion_dataset.zip`. Hỗ trợ một `fer2013.csv` gốc, hoặc thư mục `train/`, `test/`
và tùy chọn `val/` hay `validation/`.

**Không chạy hai notebook vào cùng một thư mục output.** Các tên mặc định đã được tách riêng.
Nếu thay đổi cấu hình thử nghiệm, đổi tên `RUN` để giữ lại kết quả cũ.

## 2. Dữ liệu cần có

| Nhiệm vụ | Cấu trúc hỗ trợ |
|---|---|
| Face | `classification_data/train_data/<ID>/`, `val_data/<ID>/`, `test_data/<ID>/` |
| Verification | `verification_data/*.jpg` và `verification_pairs_val.txt` ở cùng dataset root |
| Emotion CSV | Các cột `emotion`, `pixels`, `Usage`; mỗi ảnh 48x48; nhãn FER2013 gốc 0-6 |
| Emotion folders | `train/<label>/`, `test/<label>/`, tùy chọn `val/<label>/` hoặc `validation/<label>/` |

Face dataset link được người dùng xác định từ trang Kaggle:
https://www.kaggle.com/competitions/11-785-fall-20-homework-2-part-2/data

FER2013 source: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Notebook không tự đăng nhập hay tải qua tài khoản Kaggle. Người dùng tải dữ liệu được phép truy cập
rồi đưa ZIP lên Drive. Không đưa token hoặc `kaggle.json` vào repo.

## 3. Tiếp tục khi Colab ngắt kết nối

- Chạy lại phần mount Drive, giải nén và setup.
- Giữ nguyên cấu hình và dữ liệu, đặt `RESUME = True`.
- `EPOCHS` là tổng số epoch cần đạt. Nếu đã chạy 8 epoch và muốn đạt 15, giữ `EPOCHS = 15`.
- `last.pt` chứa model, optimizer, scheduler, RNG states và history của epoch hoàn thành gần nhất.
- Epoch đang chạy dở sẽ chạy lại. Không thể khôi phục chính xác giữa một batch.
- `best.pt` được chọn theo validation accuracy cho face, validation macro-F1 cho emotion.
- Có lưu seed và phiên bản thư viện, nhưng không cam kết kết quả giống từng bit giữa GPU/runtime khác nhau.
- Nếu cần đổi batch size sau lỗi thiếu GPU memory, bắt đầu một RUN mới để tránh resume sai cấu hình.

## 4. Tải kết quả về và đưa vào Git chung

ZIP kết quả có thư mục `models/` và `reports/`. **Giải nén vào thư mục gốc repo chung**.
Không đẩy weights hoặc datasets lên Git. `.gitignore` đã loại chúng ra. Chia sẻ weights qua
Drive của nhóm rồi cập nhật link thật ở phần model bên dưới.

Trên máy, khi đang ở repo đã clone:

```bash
git switch main
git pull origin main
git switch -c feature/long-classification-emotion
```

Làm bước tạo branch một lần. Sau đó chép các file source của gói này vào repo, giữ nguyên file
của người kia. Nếu repo có README hoặc `.gitignore` sẵn, ghép nội dung thay vì ghi đè toàn bộ.

Sau khi tải notebook đã chạy và report thực tế về:

```bash
git status
git add longface notebooks scripts configs docs tests requirements.txt requirements-colab.txt requirements-dev.txt
git add README.md .gitignore models/README.md reports/README.md
git add reports/
git diff --cached --stat
git commit -m "Add classification, emotion training and attendance interface"
git push -u origin feature/long-classification-emotion
```

Kiểm tra danh sách file trước khi commit. Không dùng `git add -f` cho dataset, weights, credentials
hoặc dữ liệu đăng ký. Tạo pull request để người kia review rồi merge vào `main`.
Nếu branch đã tồn tại, dùng `git switch feature/long-classification-emotion`.

Các report CSV cần thiết đã được cho phép trong `.gitignore`. `split_manifest.csv` chứa đường dẫn
ảnh và được giữ trong bản export để kiểm tra, nhưng không commit theo mặc định.

## 5. Chạy giao diện trên máy cá nhân

Dùng Python 3.11 hoặc 3.12 trong virtual environment. Trên macOS/Linux, tại repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m tkinter
```

Lệnh cuối kiểm tra Tkinter có mở cửa sổ không; đóng cửa sổ đó để tiếp tục. Nếu Python đang dùng
không có Tkinter, cài bản Python có hỗ trợ Tk/Tcl rồi tạo lại virtual environment.

Sau khi giải nén cả hai ZIP kết quả:

```bash
python -m longface.app \
  --face-model models/face_classification/best.pt \
  --thresholds reports/face_classification/verification/thresholds.json \
  --emotion-model models/emotion/best.pt \
  --demo
```

Cho phép Terminal/VS Code truy cập camera nếu macOS hỏi. Nếu camera mặc định không mở, thử `--camera 1`.
Chạy giao diện trên máy cá nhân, không chạy Tkinter trực tiếp trong hosted Colab.

### Chế độ demo và liveness

- `--demo`: cho phép thử đăng ký và ghi **log demo riêng** khi chưa có liveness; luôn có banner.
- Không có `--demo`: đăng ký và ghi điểm danh bị chặn khi liveness chưa kết nối, lỗi, uncertain hoặc spoof.
- Khi adapter có kết quả `spoof` hoặc `uncertain`, demo cũng không tự chuyển kết quả thành live.
- Thư mục `runtime/demo/` và `runtime/normal/` tách riêng gallery và log; đều bị Git bỏ qua.
- Model emotion có thể chưa truyền vào để kiểm tra phần recognition trước; UI sẽ hiển thị `not loaded`.

### Đăng ký và điểm danh

1. Đứng một mình trước camera. Nhấn **Register new person**, nhập ID và tên.
2. Hệ thống thu 10 embedding từ các ảnh đạt kiểm tra chất lượng, cách nhau ít nhất 0.5 giây.
3. Ảnh mờ, thiếu sáng, quá sáng hoặc mặt quá nhỏ sẽ bị từ chối.
4. Trung bình các embedding rồi chuẩn hóa để tạo template. Không lưu ảnh mặt thô.
5. Người đã đăng ký cần khớp ổn định qua 3 lần suy luận, đạt quality và qua liveness gate để ghi log.
6. Mỗi ID được ghi tối đa một lần mỗi ngày theo giờ máy, kể cả sau khi khởi động lại ứng dụng.
7. Người chưa khớp hiển thị `Unknown`. Nhấn nút đăng ký để thêm mới, không cần train lại CNN.
8. Nhấn **Quit**, phím `q` khi cửa sổ được focus, hoặc `Esc` để đóng và giải phóng camera.

Nếu đổi checkpoint face, phải dùng threshold tương ứng và đăng ký lại gallery. Phần mềm kiểm tra
SHA-256 để tránh trộn model, threshold và embedding khác nhau.

## 6. Innovation của Long

Registration quality gate kiểm tra độ nét bằng Laplacian variance, độ sáng trung bình và kích thước mặt.
`configs/quality.json` chứa các ngưỡng khởi đầu. Chúng là **heuristics chưa được hiệu chỉnh trên camera thật**.
Không gọi đây là model anti-spoofing hoặc khẳng định cải thiện accuracy khi chưa đo.

Đọc `docs/QUALITY_EXPERIMENT.md` để thu ảnh và đánh giá tỷ lệ nhận ảnh tốt, từ chối ảnh kém.

## 7. Ghép phần của người kia

Đọc `docs/TEAM_CONTRACT.md`. Có adapter template trong `scripts/liveness_adapter_template.py`.
Template trả `unavailable`, không giả lập kết quả live. Người kia phải ghép model đã train thật.

Ví dụ sau khi adapter được triển khai:

```bash
python -m longface.app \
  --face-model models/face_classification/best.pt \
  --thresholds reports/face_classification/verification/thresholds.json \
  --emotion-model models/emotion/best.pt \
  --liveness teammate.liveness_adapter:create_liveness \
  --liveness-checkpoint models/anti_spoofing/best.pt
```

Nếu metric-learning model tốt hơn, người kia cần viết loader/embedding adapter tương ứng theo contract.
Ứng dụng hiện tại load format `FaceClassifier` của Long; không tự nhận mọi kiến trúc metric-learning.

## 8. Đánh giá và hạn chế

- Đọc `docs/REPORT_NOTES.md` để viết phương pháp và điền kết quả thực tế.
- AUC lấy từ pairs validation theo brief. Threshold cùng bộ pairs là calibration trên validation,
  không phải bằng chứng về test độc lập hoặc mức FPR ngoài thực tế.
- Embedding L2-normalized khiến cosine và Euclidean có thứ hạng tương đương về mặt toán học.
- UI so khớp 1:N với gallery; threshold pairwise chỉ là điểm bắt đầu. Cần thử người lạ thực tế để hiệu chỉnh.
- OpenCV Haar là detector có sẵn để định vị mặt, không thay thế model nhận diện/emotion cần train.
- Detector có thể thất bại khi mặt nghiêng, bị che hoặc ánh sáng kém. Chưa hỗ trợ điểm danh nhiều người cùng lúc.
- Biểu cảm dự đoán không đủ để suy ra sức khỏe tâm thần hoặc hiệu suất làm việc.
- Trước khi nộp, cả nhóm phải tích hợp metric learning, anti-spoofing và kiểm thử bằng webcam thật.

## 9. Kiểm tra code

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Các test dùng dữ liệu tổng hợp để kiểm tra train/resume/evaluation và logic tích hợp.
Chúng không đánh giá khả năng nhận diện khuôn mặt hay chất lượng biểu cảm trên dữ liệu thật.

