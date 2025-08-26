# Đóng góp cho Library Optimizer

Cảm ơn bạn đã quan tâm đến việc đóng góp cho Library Optimizer! Tài liệu này cung cấp các hướng dẫn và chỉ dẫn để đóng góp cho dự án này.

## Quy tắc ứng xử

Vui lòng tôn trọng và quan tâm đến người khác khi đóng góp cho dự án này.

## Cách thức đóng góp

### 1. Báo cáo lỗi

Nếu bạn tìm thấy lỗi, vui lòng tạo một issue với:
- Tiêu đề rõ ràng, mô tả chi tiết
- Các bước để tái hiện lỗi
- Hành vi mong đợi
- Hành vi thực tế
- Thông tin môi trường (phiên bản Python, hệ điều hành, v.v.)
- Ví dụ mã nếu có thể

### 2. Đề xuất cải tiến

Chúng tôi hoan nghênh các đề xuất cho tính năng mới hoặc cải tiến. Vui lòng bao gồm:
- Mô tả rõ ràng về cải tiến
- Các trường hợp sử dụng và ví dụ
- Bất kỳ tài liệu tham khảo hoặc nghiên cứu liên quan

### 3. Thêm thuật toán mới

Để thêm thuật toán tối ưu hóa mới:

1. **Tuân theo mẫu**: Sử dụng mẫu trong `rules/single-objective.md`
2. **Kế thừa từ Solver**: Thuật toán của bạn phải kế thừa từ lớp cơ sở `Solver`
3. **Triển khai các phương thức bắt buộc**: Đảm bảo bạn triển khai phương thức `solver()`
4. **Thêm kiểm thử**: Bao gồm các kiểm thử toàn diện trong thư mục `test/`
5. **Cập nhật registry**: Thêm thuật toán của bạn vào registry trong `src/__init__.py`
6. **Tài liệu**: Thêm docstrings và cập nhật README nếu cần

### 4. Cải thiện tài liệu

Các cải tiến tài liệu luôn được hoan nghênh:
- Sửa lỗi chính tả và ngữ pháp
- Cải thiện tính rõ ràng và các ví dụ
- Thêm tài liệu còn thiếu
- Dịch tài liệu (nếu áp dụng)

## Thiết lập môi trường phát triển

1. **Fork repository**
2. **Clone fork của bạn**:
   ```bash
   git clone https://github.com/your-username/LibraryOptimizer.git
   cd LibraryOptimizer
   ```

3. **Cài đặt các phụ thuộc phát triển**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Tạo một nhánh**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Tiêu chuẩn mã nguồn

### Hướng dẫn phong cách

- Tuân theo hướng dẫn phong cách PEP 8
- Sử dụng 4 khoảng trắng cho thụt lề
- Độ dài dòng tối đa: 88 ký tự
- Sử dụng type hints cho tất cả các hàm public
- Viết docstrings toàn diện

### Quy ước đặt tên

- **Lớp**: PascalCase (ví dụ: `GreyWolfOptimizer`)
- **Hàm/Phương thức**: snake_case (ví dụ: `create_solver`)
- **Biến**: snake_case (ví dụ: `objective_func`)
- **Hằng số**: UPPER_SNAKE_CASE (ví dụ: `MAX_ITERATIONS`)

### Tài liệu

Tất cả các hàm và lớp public phải có docstrings theo định dạng này:

```python
def function_name(param1: Type, param2: Type = default) -> ReturnType:
    """
    Mô tả ngắn gọn về hàm.
    
    Tham số:
    -----------
    param1 : Type
        Mô tả param1
    param2 : Type, optional
        Mô tả param2, mặc định là default
        
    Trả về:
    --------
    ReturnType
        Mô tả giá trị trả về
        
    Ngoại lệ:
    -------
    ValueError
        Khi có lỗi xảy ra
    """
```

## Kiểm thử

### Viết kiểm thử

- Đặt các kiểm thử trong thư mục `test/`
- Sử dụng tên hàm kiểm thử mô tả bắt đầu bằng `test_`
- Kiểm thử cả bài toán tối thiểu hóa và tối đa hóa
- Bao gồm kiểm thử cho các trường hợp biên
- Sử dụng các hàm benchmark tiêu chuẩn (Sphere, Rastrigin, v.v.)

### Chạy kiểm thử

```bash
# Chạy tất cả kiểm thử
python -m pytest test/ -v

# Chạy file kiểm thử cụ thể
python -m pytest test/test-gwo.py -v

# Chạy với coverage
python -m pytest test/ --cov=src --cov-report=html
```

### Chất lượng mã nguồn

```bash
# Chạy flake8 để kiểm tra chất lượng mã
flake8 src/ --max-line-length=120

# Chạy black để định dạng mã
black src/

# Chạy isort để sắp xếp import
isort src/

# Chạy mypy để kiểm tra kiểu
mypy src/ --ignore-missing-imports
```

## Quy trình Pull Request

1. **Đảm bảo kiểm thử pass**: Tất cả kiểm thử phải pass trước khi gửi PR
2. **Cập nhật tài liệu**: Bao gồm các cập nhật tài liệu liên quan
3. **Tuân theo mẫu**: Sử dụng mẫu PR nếu có sẵn
4. **Mô tả thay đổi**: Cung cấp mô tả rõ ràng về các thay đổi của bạn
5. **Tham chiếu issues**: Liên kết đến các issues liên quan

## Quy trình đánh giá

- PR sẽ được đánh giá bởi các maintainer
- Phản hồi có thể được cung cấp để cải tiến
- Khi được chấp thuận, PR của bạn sẽ được merge

## Giấy phép

Bằng cách đóng góp cho dự án này, bạn đồng ý rằng các đóng góp của bạn sẽ được cấp phép theo Giấy phép MIT.

## Có câu hỏi?

Nếu bạn có câu hỏi về việc đóng góp, vui lòng:
1. Kiểm tra tài liệu hiện có
2. Tìm kiếm các issues hiện có
3. Tạo issue mới nếu câu hỏi của bạn chưa được trả lời

Cảm ơn bạn đã đóng góp cho Library Optimizer!