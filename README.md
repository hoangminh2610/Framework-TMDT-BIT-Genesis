# E-commerce customer behavior dashboard

Dashboard nghiên cứu để khám phá dữ liệu sự kiện TMĐT, xây dựng RFM và đặc trưng hành vi, rồi so sánh ba cấu hình phân cụm.

## Chạy ứng dụng

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Mặc định ứng dụng dùng dữ liệu tổng hợp có các kiểu hành vi minh họa. Đây không phải dữ liệu REES46 và không dùng để tái tạo kết quả thực nghiệm trong luận văn. Để phân tích dữ liệu thật, tải CSV REES46 có các cột `event_time`, `event_type`, `product_id`, `price`, `user_id` và `user_session`.

## ETL với Spark local

Để xử lý event log lớn hơn bằng Spark trên máy hiện tại:

```powershell
python -m pip install -r requirements-spark.txt
```

Cần cài Java tương thích với PySpark 3.5 và cấu hình `JAVA_HOME`. Trong ứng dụng, chọn **Spark local[*]**. Đây là Spark chạy trên một máy, chưa phải cụm phân tán. Kết quả feature vẫn được đưa vào scikit-learn cục bộ; K-Means chạy trên toàn bộ khách hàng hợp lệ, DBSCAN chạy trên mẫu phân tầng tối đa 50.000 khách hàng.

Chế độ Pandas phù hợp với tệp nhỏ hơn. Với bộ REES46 đầy đủ, dùng Spark local nếu máy đủ RAM/đĩa; giới hạn upload của Streamlit cũng có thể cần được cấu hình theo kích thước tệp.

## Cách đọc kết quả

- C0 và C1 thay đổi bộ đặc trưng; C1 và C2 thay đổi thuật toán.
- Các chỉ số C0/C1 được ước lượng trên tối đa 5.000 khách hàng để giảm thời gian tính Silhouette. DBSCAN được đánh giá riêng trên mẫu đã chạy.
- Nhãn phân khúc và cờ Recency là gợi ý diễn giải, không phải dự báo churn. CLV chưa được tính.
- PCA là phép chiếu để xem cụm; nó không thay thế các chỉ số benchmark trong không gian đặc trưng.
