# Sơ đồ thuật toán Multi Objective Modified Social Group Optimizer

```mermaid
flowchart TD
    A["Bắt đầu"] --> B["Khởi tạo quần thể"]
    B --> C["Khởi tạo archive với giải pháp không bị chi phối"]
    C --> D["Tạo grid cho archive"]
    D --> E["Bắt đầu vòng lặp chính"]
    E --> F["Phase 1: Guru Phase"]
    F --> G["Chọn guru từ archive"]
    G --> H["Cập nhật vị trí: c*current + rand*(guru - current)"]
    H --> I["Kiểm tra biên và cập nhật fitness"]
    I --> J["Lựa chọn tham lam: chấp nhận nếu tốt hơn"]
    J --> K["Phase 2: Learner Phase"]
    K --> L["Chọn global best từ archive"]
    L --> M["Chọn partner ngẫu nhiên"]
    M --> N{"Kiểm tra quan hệ chi phối"}
    N -- "Current dominates partner" --> O{"Xác suất khám phá ngẫu nhiên?"}
    N -- "Partner dominates current hoặc non-dominated" --> P["Học từ partner: current + rand*(partner - current) + rand*(global_best - current)"]
    O -- "Cao" --> Q["Khám phá ngẫu nhiên"]
    O -- "Thấp" --> R["Học từ partner: current + rand*(current - partner) + rand*(global_best - current)"]
    Q --> S["Kiểm tra biên và cập nhật fitness"]
    R --> S
    P --> S
    S --> T["Lựa chọn tham lam: chấp nhận nếu tốt hơn"]
    T --> U["Cập nhật archive"]
    U --> V["Lưu trữ lịch sử archive"]
    V --> W{"iter >= max_iter?"}
    W -- "Chưa" --> E
    W -- "Rồi" --> X["Kết thúc"]
    
    subgraph Khởi tạo
    B
    C
    D
    end
    
    subgraph Vòng lặp chính
    F
    G
    H
    I
    J
    K
    L
    M
    N
    O
    P
    Q
    R
    S
    T
    U
    V
    end
```

### Giải thích chi tiết các bước:

1. **Khởi tạo quần thể**: 
   - Tạo ngẫu nhiên các vị trí ban đầu trong không gian tìm kiếm
   - Mỗi vị trí X_i ∈ [lb, ub]^dim
   - Tính toán giá trị hàm mục tiêu multi_fitness = objective_func(X_i)

2. **Khởi tạo archive với giải pháp không bị chi phối**:
   - Xác định các giải pháp không bị chi phối trong quần thể ban đầu
   - Thêm các giải pháp này vào archive bên ngoài

3. **Tạo grid cho archive**:
   - Tạo lưới hypercubes để quản lý archive
   - Gán chỉ số grid cho từng giải pháp trong archive

4. **Vòng lặp chính** (max_iter lần):
   - **Phase 1: Guru Phase** (học từ người giỏi nhất):
     * Chọn guru từ archive sử dụng grid-based selection
     * Cập nhật vị trí:
       ```python
       new_position[j] = c * current.position[j] + random * (guru.position[j] - current.position[j])
       ```
     * Kiểm tra biên và cập nhật fitness
     * Lựa chọn tham lam: chấp nhận giải pháp mới nếu nó chi phối giải pháp hiện tại hoặc không bị chi phối
   
   - **Phase 2: Learner Phase** (học lẫn nhau với khám phá ngẫu nhiên):
     * Chọn global best từ archive
     * Chọn partner ngẫu nhiên khác với cá thể hiện tại
     * Kiểm tra quan hệ chi phối giữa current và partner:
       - Nếu current dominates partner và không bị partner dominates:
         * Nếu random > sap (0.7): học từ partner
           ```python
           new_position = current + random*(current - partner) + random*(global_best - current)
           ```
         * Ngược lại: khám phá ngẫu nhiên
       - Ngược lại (partner dominates current hoặc non-dominated): học từ partner
         ```python
         new_position = current + random*(partner - current) + random*(global_best - current)
         ```
     * Kiểm tra biên và cập nhật fitness
     * Lựa chọn tham lam: chấp nhận giải pháp mới nếu tốt hơn
   
   - **Cập nhật archive**: Thêm các giải pháp không bị chi phối mới vào archive
   
   - **Lưu trữ lịch sử archive**: Lưu trạng thái archive hiện tại

5. **Kết thúc**:
   - Lưu trữ kết quả cuối cùng
   - Trả về lịch sử archive và archive cuối cùng
