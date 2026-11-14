# Báo cáo cá nhân — Day 6

- **Họ và tên:** Nguyễn Tấn Châu
- **MSSV:** 2A202603846
- **Lớp:** L3B

## Dữ liệu và pipeline
- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:**
  - Chủ đề: Tài liệu kỹ thuật — *Prompt Engineering* (Google Whitepaper, tác giả Lee Boonstra, phiên bản February 2025).
  - Nguồn & quyền: `data/raw/Prompt Engineering.pdf` (tài liệu công khai của Google phục vụ mục đích học tập/nghiên cứu; chi tiết tại [data/manifest.json](../../data/manifest.json)).
  - Số trang: **68 trang vật lý**, giữ lại **62 trang nội dung** (trang 6–67, tổng `77.336` ký tự) và loại **6 trang phi nội dung** (trang 1 bìa, trang 2 Acknowledgements, trang 3–5 mục lục, trang 68 Endnotes) ghi nhận tại [results/extraction_log.json](../../results/extraction_log.json).
- **Chuẩn hóa, lỗi và giới hạn:**
  - Trích xuất bằng `pymupdf` (`fitz`), cắt margin trên `9%` và margin dưới `10%` để loại bỏ triệt để header lặp `"Prompt Engineering"` và footer `"February 2025 | <số trang>"`.
  - Chuẩn hóa Unicode `NFC`, xóa ký tự điều khiển (`\x08`), chuẩn hóa khoảng trắng và giữ cấu trúc đoạn (`\n\n`). Không phát hiện trang lỗi/scan (`errors: []`).
- **Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí:**
  - Embedding model: `BAAI/bge-m3` qua `HuggingFace / sentence-transformers` (local), `1024` chiều, `normalize_embeddings=True`.
  - Vector store: `ChromaDB` (`1.5.9`), lưu bền vững tại `data/chroma_db`, metric `cosine` (`hnsw:space: cosine`), tìm kiếm HNSW approximate nearest neighbor.
  - Chi phí: `0.0 USD` (chạy hoàn toàn local).
  - Lệnh chạy tái lập: xem [README.md](../../README.md).
- **Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn:**
  - Bắt buộc: `document_id`, `source`, `page` (số trang PDF vật lý).
  - Bổ sung: `section` (kiểu `string`, trích xuất tự động từ TOC/tiêu đề chương gồm: `Introduction`, `LLM Output Configuration`, `Prompting Techniques`, `Code Prompting`, `Best Practices`, `Summary`) và `year` (`2025`). Chọn `section` làm trường lọc vì tài liệu chia chương rõ ràng theo chủ đề kỹ thuật, giúp thu hẹp phạm vi truy xuất khi câu hỏi chỉ định ngữ cảnh chương.

## Cấu hình và benchmark
- **A / B / C (tham số cụ thể, khác biệt duy nhất B–C):**
  - **Strategy A:** Fixed-size `500` ký tự, overlap `50` ký tự; metadata đầy đủ; **không** dùng filter khi truy xuất.
  - **Strategy B:** Structure-aware theo heading/đoạn văn, giới hạn tối đa `800` ký tự; metadata giống A; **không** dùng filter khi truy xuất.
  - **Strategy C:** Dùng chung bộ chunk và vector của Strategy B; khác biệt duy nhất là áp dụng metadata filter (`where={"section": ...}`) khi câu hỏi có `filter != null`.
- **≥20 câu: phân chia ≥15 dev / ≥5 test, câu metadata filter và cách kiểm chứng evidence:**
  - Bộ [benchmark/questions.json](../../benchmark/questions.json) gồm **20 câu**: **15 `dev`** (`q01`–`q15`) và **5 `test`** (`q16`–`q20`).
  - Phân loại: `truc_tiep` (9), `dieu_kien_ngoai_le` (6), `dien_dat_lai` (5).
  - Câu có metadata filter (`section`): **5 câu** (3 `dev`: `q04`, `q08`, `q13`; 2 `test`: `q18`, `q19`), sử dụng giá trị `section` làm **input ngữ cảnh giả định được cung cấp trước khi truy xuất** (không suy diễn từ gold source).
  - Kiểm chứng độc lập bằng [src/verify_benchmark.py](../../src/verify_benchmark.py): 100% `evidence_quote` khớp nguyên văn tại `gold_page` trong [data/processed/pages.jsonl](../../data/processed/pages.jsonl) và 100% giá trị `filter` khớp metadata thực tế.
- **Tiêu chí chọn chiến lược đã chốt trước khi xem test:**
  - Ưu tiên `Evidence Hit@3` cao nhất, tiếp theo là `Grounding Quality` & `Chunk Coherence`, sau đó là tính an toàn phạm vi truy xuất (filter) và tài nguyên ingest (số chunk, thời gian).

## Kết quả

### Bảng 1: Hiệu năng truy xuất và tài nguyên (số liệu tự động)
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian ingest | Chi phí |
|---|---|---|---|---:|---:|---:|
| A | dev | 9/15 (60.0%) | 13/15 (86.67%) | 196 | 466.146s | \$0.0 |
| A | test | 4/5 (80.0%) | 4/5 (80.0%) | 196 | 466.146s | \$0.0 |
| B | dev | 11/15 (73.33%) | 14/15 (93.33%) | 165 | 413.137s | \$0.0 |
| B | test | 5/5 (100.0%) | 5/5 (100.0%) | 165 | 413.137s | \$0.0 |
| C | dev | 11/15 (73.33%) | 14/15 (93.33%) | 165 | 2.344s (reused B) | \$0.0 |
| C | test | 5/5 (100.0%) | 5/5 (100.0%) | 165 | 2.344s (reused B) | \$0.0 |

### Bảng 2: Đánh giá thủ công (Coherence, Metadata, Grounding)
> *Ghi chú: Điểm chấm thủ công được tổng hợp từ [coherence_audit.csv](../../results/manual_audit/coherence_audit.csv) (10 mẫu đầu `p006`–`p015`) và [grounding_and_metadata_audit.csv](../../results/manual_audit/grounding_and_metadata_audit.csv) (5 câu `test` × top-3). Chiến lược B và C dùng chung bộ chunk nên điểm Coherence bằng nhau.*

| Strategy | Coherence trung bình /2 (n=10) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| A | **0.90 / 2** (0/1/2: 2/7/1) | 15/15 (100%) | N/A | 0 / 1 / 4 — **1.80 / 2** |
| B | **1.40 / 2** (0/1/2: 1/4/5) | 15/15 (100%) | N/A | 0 / 0 / 5 — **2.00 / 2** |
| C | **1.40 / 2** (0/1/2: 1/4/5) | 15/15 (100%) | Test: 2/2 (Dev: 3/3, Tổng: 5/5 = 100%) | 0 / 0 / 5 — **2.00 / 2** |

- **Đường dẫn trace, nhãn thủ công và cách chọn mẫu:**
  - Traces: [trace_A.json](../../results/traces/trace_A.json), [trace_B.json](../../results/traces/trace_B.json), [trace_C.json](../../results/traces/trace_C.json).
  - Tổng hợp chỉ số: [metrics_summary.json](../../results/metrics_summary.json).
  - Kết quả chấm thủ công: [coherence_audit.csv](../../results/manual_audit/coherence_audit.csv) (10 page ID đầu tiên đã sắp xếp: `p006`–`p015`) và [grounding_and_metadata_audit.csv](../../results/manual_audit/grounding_and_metadata_audit.csv) (5 câu `test` × top-3 = 45 bản ghi).

## Phân tích và kết luận
- **So sánh A–B và B–C; 2 ví dụ sai có ID câu và nguồn:**
  - **So sánh A–B (tác động của chunking):** Structure-aware chunking (B) cải thiện rõ rệt so với fixed-size 500 ký tự (A): `Hit@1` trên `dev` tăng từ **60.0% (9/15)** lên **73.33% (11/15)**, `Evidence Hit@3` trên `dev` tăng từ **86.67% (13/15)** lên **93.33% (14/15)**; trên `test`, `Evidence Hit@3` tăng từ **80.0% (4/5)** lên **100% (5/5)**. Coherence tăng mạnh từ **0.90/2** lên **1.40/2** và Grounding tăng từ **1.80/2** lên **2.00/2**. Đồng thời, số lượng chunk giảm từ **196** xuống **165**, giúp giảm ~11.4% thời gian embedding từ **466.15s** xuống **413.14s**.
  - **So sánh B–C (tác động của metadata filter):** C giữ nguyên chất lượng retrieval của B (`dev`: Hit@1 73.33%, Hit@3 93.33%; `test`: Hit@1 100%, Hit@3 100%; Grounding 2.00/2) và đạt **100% Filter Compliance** (`3/3` trên `dev`, `2/2` trên `test`, tổng `5/5`). Việc lọc theo `section` giúp cô lập không gian tìm kiếm đúng chương mục tiêu (ví dụ ở câu `q18`, cả top-3 chunk của C đều nằm trong `Best Practices`, loại bỏ hoàn toàn các chunk nhiễu về `few-shot` ở chương `Prompting Techniques` như ở B), giảm nguy cơ nhiễu chéo chương khi mở rộng corpus nhiều tài liệu.
  - **2 ví dụ lỗi cụ thể:**
    1. *Lỗi cắt cứng ở Strategy A — câu `q19` (nguồn: `Prompt Engineering.pdf`, trang 62):* Do cắt cố định 500 ký tự, câu bằng chứng về `JSON Schema` bị xé đôi sang 2 chunk khác nhau (`rank 1` chỉ giữ phần đầu, `rank 3` giữ vế sau nhờ overlap 50 ký tự), khiến cả `Hit@1` và `Evidence Hit@3` của A đều trượt và Grounding chỉ đạt `1/2`; trong khi B và C giữ trọn đoạn văn nên đạt `Hit@1` và Grounding `2/2`.
    2. *Hạn chế của Strategy B/C — vị trí `p015` (nguồn: `Prompt Engineering.pdf`, trang 15, Coherence = `0/2`):* Do tách theo dấu xuống dòng và gom nhóm, phần đầu bảng `Table 1` (`Name`, `Goal`, `Model`...) bị tách rời khỏi nội dung `Prompt`/`Output` ở chunk tiếp theo, làm mất ngữ cảnh bảng biểu.
- **Chiến lược chọn, đánh đổi, điều kiện áp dụng:**
  - **Đề xuất chọn Chiến lược C** (Structure-aware chunking + Metadata filter `section`) làm cấu hình tối ưu vì đạt cao nhất ở mọi tiêu chí đã chốt trước (`Evidence Hit@3`, `Grounding 2.00/2`, `Coherence 1.40/2`, `Filter Compliance 100%`) với chi phí lưu trữ/tính toán thấp hơn A.
  - *Đánh đổi & điều kiện áp dụng:* B/C có chunk tối đa 800 ký tự nên ngữ cảnh rộng hơn nhưng có thể chứa thêm câu dẫn phụ; C phát huy tác dụng rõ nhất khi ngữ cảnh truy vấn xác định được `section` hợp lệ trước khi truy xuất (nếu người dùng không cung cấp hoặc đoán sai `section`, filter cứng sẽ làm sót kết quả).
- **Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí:**
  - Tập `test` chỉ có **5 câu** (và chỉ 2 câu có filter) nên chỉ mang tính minh họa cho pipeline kiểm chứng, chưa đủ cỡ mẫu để suy rộng kết luận tổng quát ngoài corpus này.
  - Việc chấm thủ công Coherence và Grounding do chính tác giả thực hiện nên có nguy cơ thiên lệch xác nhận (confirmation bias).
  - Corpus chỉ gồm 1 tài liệu PDF 62 trang nội dung nên độ phân tách từ filter `section` chưa tạo chênh lệch điểm số lớn giữa B và C như khi chạy trên kho đa tài liệu.
  - Có sử dụng AI hỗ trợ viết code pipeline (`extract_and_clean.py`, `verify_benchmark.py`, `ingest.py`, `evaluate.py`) và rà soát đối chiếu số liệu; mọi kết quả định lượng và trace đều được chạy kiểm chứng trực tiếp trên dữ liệu thật.
- **Nếu làm lại hoặc mở rộng, cải tiến gì?**
  - Bổ sung nhận diện khối bảng/code block trong bước chunking (giữ trọn bảng như `Table 1` ở trang 15 thay vì tách theo dòng đơn lẻ).
  - Mở rộng benchmark lên nhiều tài liệu cùng chủ đề (ví dụ các whitepaper khác về Agents, Embeddings) và thử nghiệm thêm **hybrid retrieval** (kết hợp dense `BAAI/bge-m3` + BM25) cho các truy vấn chứa tên hàm/tham số chính xác.
