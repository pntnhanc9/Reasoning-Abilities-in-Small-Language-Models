# 🚀 GRPO Training & Evaluation (DeepSeek & Qwen)

Dự án này chứa mã nguồn, dữ liệu kết quả và các công cụ (notebook) để huấn luyện (sử dụng thuật toán GRPO) và đánh giá các mô hình ngôn ngữ lớn (LLMs) như **DeepSeek** và **Qwen** trên nền tảng Kaggle.

## 📁 Cấu trúc thư mục
```text
GRPO/
│
├── Kaggle_data/                  # Thư mục chứa dữ liệu và kết quả huấn luyện từ Kaggle
│   ├── results_deepseek_1/       # Kết quả & source code cho DeepSeek (Run 1)
│   │   └── grpo-source-code/     # ↳ Thư mục chứa Notebook training gửi lên Kaggle
│   ├── results_deepseek_2/       # Kết quả & source code cho DeepSeek (Run 2)
│   │   └── grpo-source-code/
│   ├── results_qwen_1/           # Kết quả & source code cho Qwen (Run 1)
│   │   └── grpo-source-code/
│   └── results_qwen_2/           # Kết quả & source code cho Qwen (Run 2)
│       └── grpo-source-code/
│
├── compare_kaggle_models.ipynb   # Notebook so sánh hiệu năng giữa các mô hình/lần chạy
└── evaluation  # Folder đánh giá chi tiết kết quả của từng mô hình của tác giả trên MultiArith và SVAMP
## 🛠️ Hướng dẫn Huấn luyện (Training) trên Kaggle

Để chạy lại quá trình huấn luyện mô hình trên Kaggle, hãy thực hiện theo các bước sau:

1. **Chọn mô hình muốn huấn luyện:** 
   Truy cập vào thư mục `Kaggle_data` và chọn 1 trong 4 thư mục kết quả mà bạn muốn chạy (ví dụ: `results_qwen_1`).
2. **Lấy mã nguồn:** 
   Bên trong thư mục bạn vừa chọn, mở tiếp thư mục `grpo-source-code`. Tại đây sẽ có chứa file Notebook huấn luyện (`.ipynb`).
3. **Đưa mã nguồn lên Kaggle:**
   * Tạo một Notebook mới trên Kaggle (hoặc tạo Dataset chứa thư mục `grpo-source-code` rồi import vào Notebook).
   * Upload file Notebook từ máy của bạn lên môi trường Kaggle.
4. **Cấu hình đường dẫn (Links/Paths):**
   * Mở Notebook trên Kaggle.
   * Kiểm tra và chỉnh sửa lại các đường dẫn dữ liệu (links/paths) trong Notebook sao cho khớp với môi trường Kaggle hiện tại (đường dẫn tới dataset, thư mục lưu output, v.v.).
5. **Thực thi:**
   * Bật GPU phù hợp (khuyến nghị P100, T4x2 hoặc GPU mạnh hơn tùy cấu hình).
   * Bấm **Run All** để bắt đầu quá trình huấn luyện mô hình với GRPO.

---

## 📊 Hướng dẫn So sánh các model được fine tune và train (Comparison)

Sau khi có kết quả tải về từ Kaggle (lưu vào các thư mục `results_*`), bạn sử dụng 2 file Notebook ở thư mục gốc để phân tích:

* 📈 **`compare_kaggle_models.ipynb`**: Sử dụng file này để vẽ biểu đồ và so sánh trực quan hiệu năng giữa các lần chạy khác nhau (DeepSeek 1 vs 2, Qwen 1 vs 2, hoặc DeepSeek vs Qwen).

> **Lưu ý:** Trước khi chạy 2 notebook này ở máy local, hãy đảm bảo bạn đã cài đặt đủ các thư viện cần thiết (`pandas`, `matplotlib`, `seaborn`, `jupyter`,...) và trỏ đúng đường dẫn đọc dữ liệu vào thư mục `Kaggle_data`.

---

## 📝 Hướng dẫn Đánh giá mô hình (Evaluation)

Để đánh giá chi tiết hiệu năng của các mô hình Open-RS, bạn sẽ sử dụng các file nằm trong thư mục `GRPO/evaluation/`. Bạn cần tải các file notebook và tập dữ liệu đánh giá lên nền tảng Kaggle để tiến hành chạy thử nghiệm:

* 📂 **Dữ liệu đánh giá (Datasets):** Hai file `MultiArith_eval_ready.jsonl` và `SVAMP_eval_ready.jsonl` chứa các câu hỏi toán học đã được chuẩn bị sẵn.
* 📓 **Notebook chạy đánh giá:** Bao gồm 6 file `.ipynb` tương ứng với 3 phiên bản mô hình (Open-RS1, Open-RS2, Open-RS3) được test độc lập trên 2 tập dữ liệu (MultiArith và SVAMP). Ví dụ: `Open-RS1-MultiArith.ipynb`, `Open-RS2-SVAMP.ipynb`,... 

> **Lưu ý:** Khi đưa các file này lên Kaggle để chạy thực nghiệm, hãy đảm bảo bạn đã upload các file dữ liệu `.jsonl` lên Kaggle (hoặc Add Data vào môi trường làm việc). Đồng thời, nhớ kiểm tra và trích xuất đúng đường dẫn thư mục (ví dụ: `/kaggle/input/...`) vào trong code đọc dữ liệu của từng file Jupyter Notebook trước khi bấm chạy!

---

## 📌 Yêu cầu hệ thống (Requirements)

* **Ngôn ngữ:** Python 3.8+
* **HF_TOKEN:** lấy token trên hugging face và đặt vào secret keys của Kaggle.
* **Môi trường:** Kaggle (cho quá trình Training và Evaluation) & Local Jupyter Notebook (cho quá trình Evaluation).
* **Các thư viện chính:** `transformers`, `trl`, `peft`, `torch`, `pandas`, `matplotlib`. đã được cày trong notebook, có thể chạy trên môi trường Kaggle
