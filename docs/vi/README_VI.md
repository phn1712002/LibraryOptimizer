# Library Optimizer

Một thư viện Python toàn diện cho các thuật toán tối ưu metaheuristic, tích hợp hơn 20 thuật toán tối ưu tiên tiến với khả năng tối ưu đơn mục tiêu và đa mục tiêu, tất cả đều có giao diện thống nhất.

## Tính năng

- **Hơn 20 Thuật Toán Tối Ưu**: Grey Wolf Optimizer, Particle Swarm Optimization, Artificial Bee Colony, Whale Optimization, và nhiều thuật toán khác
- **Hỗ trợ Đa Mục Tiêu**: Tự động tạo phiên bản đa mục tiêu từ các thuật toán đơn mục tiêu
- **Giao Diện Thống Nhất**: API nhất quán trên tất cả các thuật toán cho cả tối ưu đơn và đa mục tiêu
- **Trực Quan Hóa**: Theo dõi tiến trình và vẽ biểu đồ hội tụ tích hợp sẵn
- **Hàm Benchmark**: Các hàm kiểm tra sẵn sàng sử dụng để đánh giá (đơn và đa mục tiêu)
- **Mở Rộng**: Dễ dàng thêm thuật toán mới theo các mẫu có sẵn
- **Gợi Ý Kiểu**: Đầy đủ chú thích kiểu dữ liệu để trải nghiệm phát triển tốt hơn
- **Tự Động Phát Hiện**: Hệ thống tự động phát hiện loại hàm mục tiêu và chọn bộ giải phù hợp

## Cài đặt

```bash
git clone https://github.com/phn1712002/LibraryOptimizer
cd LibraryOptimizer && pip install -e . 
```

## Bắt Đầu Nhanh

### Tối Ưu Đơn Mục Tiêu

```python
import numpy as np
from LibraryOptimizer import create_solver

# Định nghĩa hàm mục tiêu (Hàm Sphere - đơn mục tiêu)
def sphere_function(x):
    return np.sum(x**2)

# Tạo đối tượng tối ưu
optimizer = create_solver(
    solver_name='GreyWolfOptimizer',
    objective_func=sphere_function,
    lb=-5.0,  # Giới hạn dưới
    ub=5.0,   # Giới hạn trên
    dim=10,   # Số chiều bài toán
    maximize=False  # Bài toán tối thiểu hóa
)

# Chạy tối ưu
history, best_solution = optimizer.solver(
    search_agents_no=50,  # Kích thước quần thể
    max_iter=100          # Số lần lặp tối đa
)

print(f"Giải pháp tốt nhất: {best_solution.position}")
print(f"Giá trị fitness tốt nhất: {best_solution.fitness}")
```

### Tối Ưu Đa Mục Tiêu

```python
import numpy as np
from LibraryOptimizer import create_solver

# Định nghĩa hàm đa mục tiêu (Benchmark ZDT1)
def zdt1_function(x):
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2])

# Tạo đối tượng tối ưu - hệ thống tự động phát hiện hàm đa mục tiêu
optimizer = create_solver(
    solver_name='GreyWolfOptimizer',  # Cùng tên, tự động phát hiện đa mục tiêu
    objective_func=zdt1_function,
    lb=np.array([0.0, 0.0]),  # Giới hạn dưới
    ub=np.array([1.0, 1.0]),  # Giới hạn trên
    dim=2,                    # Số chiều bài toán
    archive_size=100          # Kích thước archive cho tối ưu đa mục tiêu
)

# Chạy tối ưu
history_archive, final_archive = optimizer.solver(
    search_agents_no=100,     # Kích thước quần thể
    max_iter=200              # Số lần lặp tối đa
)

print(f"Tìm thấy {len(final_archive)} giải pháp không bị chi phối")
print(f"Giải pháp đầu tiên: {final_archive[0].position} -> {final_archive[0].multi_fitness}")
```

## Các Thuật Toán Có Sẵn

Tất cả các thuật toán đều có sẵn ở cả phiên bản đơn mục tiêu và đa mục tiêu:

- Grey Wolf Optimizer (GWO) / Multi-Objective GWO
- Whale Optimization Algorithm (WOA) / Multi-Objective WOA
- Particle Swarm Optimization (PSO) / Multi-Objective PSO
- Artificial Bee Colony (ABC) / Multi-Objective ABC
- Ant Colony Optimization (ACO) / Multi-Objective ACO
- Bat Algorithm / Multi-Objective Bat
- Artificial Ecosystem-based Optimization (AEO) / Multi-Objective AEO
- Cuckoo Search (CS) / Multi-Objective CS
- Dingo Optimization Algorithm (DOA) / Multi-Objective DOA
- Firefly Algorithm / Multi-Objective Firefly
- JAYA Algorithm / Multi-Objective JAYA
- Modified Social Group Optimization (MSGO) / Multi-Objective MSGO
- Moss Growth Optimization (MGO) / Multi-Objective MGO
- Shuffled Frog Leaping Algorithm (SFLA) / Multi-Objective SFLA
- Teaching-Learning-based Optimization (TLBO) / Multi-Objective TLBO
- Prairie Dogs Optimization (PDO) / Multi-Objective PDO
- Simulated Annealing (SA) / Multi-Objective SA
- Genetic Algorithm (GA) / Multi-Objective GA
- Và nhiều hơn nữa....

Hệ thống tự động chọn phiên bản phù hợp dựa trên loại hàm mục tiêu của bạn.

## Tài Liệu

Tài liệu đầy đủ có sẵn tại [Kho Lưu Trữ GitHub](https://github.com/phn1712002/LibraryOptimizer).

## Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng xem [Hướng Dẫn Đóng Góp](CONTRIBUTING_VI.md) để biết chi tiết.

### Thêm Thuật Toán Mới

Để thêm một thuật toán tối ưu mới, hãy làm theo mẫu trong `rules/single-objective.md`.
Nếu bạn cần triển khai nhiều fitness, hãy làm theo mẫu trong `rules/multi-objective.md`.
Hoặc hệ thống sẽ tự động tạo phiên bản đa mục tiêu của thuật toán của bạn bằng hệ thống tạo tự động được mô tả trong `rules/auto-generation-multi.md`.

### Kịch Bản Kiểm Thử Mẫu

Thư viện bao gồm một kịch bản kiểm thử toàn diện (`test/example_test.py`) cho phép bạn kiểm tra các thuật toán tối ưu cụ thể với các hàm benchmark khác nhau:

```bash
# Hiển thị các thuật toán có sẵn
python test/example_test.py -list

# Kiểm tra một thuật toán cụ thể (ví dụ: GreyWolfOptimizer)
python test/example_test.py -name GreyWolfOptimizer

# Kiểm tra một thuật toán khác (ví dụ: ParticleSwarmOptimizer)  
python test/example_test.py -name ParticleSwarmOptimizer
```

Kịch bản kiểm thử chạy các bài kiểm tra toàn diện sau:
- **Hàm Sphere**: Kiểm tra khả năng tối thiểu hóa cơ bản
- **Hàm Rastrigin**: Kiểm tra hiệu suất trên hàm đa mode
- **Tối Đa Hóa**: Kiểm tra khả năng tối đa hóa hàm của thuật toán
- **Đa Mục Tiêu ZDT1**: Kiểm tra tối ưu đa mục tiêu với 2 mục tiêu
- **Đa Mục Tiêu ZDT5**: Kiểm tra tối ưu đa mục tiêu với 3 mục tiêu

Mỗi bài kiểm tra xác nhận rằng thuật toán có thể tìm ra các giải pháp hợp lý và cung cấp báo cáo chi tiết đạt/không đạt.

## Giấy Phép

Giấy phép MIT - xem tệp [LICENSE](LICENSE) để biết chi tiết.

## Trích Dẫn

Nếu bạn sử dụng thư viện này trong nghiên cứu của mình, vui lòng cân nhắc trích dẫn:

```bibtex
@software{LibraryOptimizer,
  author = {phn1712002},
  title = {Library Optimizer: A Python Library for Metaheuristic Optimization with Multi-Objective Support},
  year = {2025},
  url = {https://github.com/phn1712002/LibraryOptimizer}
}
```

## Liên Hệ

- Tác giả: phn1712002
- Email: phn1712002@gmail.com
- GitHub: [phn1712002](https://github.com/phn1712002)

## Lời Cảm Ơn

Thư viện này được xây dựng dựa trên nghiên cứu về tối ưu metaheuristic và triển khai các thuật toán từ nhiều ấn phẩm khoa học khác nhau.