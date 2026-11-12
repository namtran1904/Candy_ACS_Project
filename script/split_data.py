import os
import random
import shutil

#thiết lập đường dẫn
RAW_DIR = os.path.join("data", "raw_data_set")
TRAIN_DIR = os.path.join("data", "data_set", "train_data_set")
TEST_DIR = os.path.join("data", "data_set", "test_data_set")

#tỉ lệ phân tách
SPLIT_RATIO = 0.8

#lấy danh sách các nhãn trong raw
danh_sach_keo = os.listdir(RAW_DIR)

#với mỗi nhãn trong danh sách các nhãn raw
for loai_keo in danh_sach_keo:
    #tạo đường dẫn thư mục train và test đích cho từng nhãn
    duong_dan_train_folder = os.path.join(TRAIN_DIR,loai_keo)
    duong_dan_test_folder = os.path.join(TEST_DIR, loai_keo)

    #tạo thư mục train và test 
    os.makedirs(duong_dan_train_folder, exist_ok = True)
    os.makedirs(duong_dan_test_folder, exist_ok = True)

    #lấy đường dẫn trỏ đến thư mục raw của từng nhãn
    raw_folder = os.path.join(RAW_DIR,loai_keo);
    danh_sach_anh = os.listdir(raw_folder)

    #xáo trộn ảnh
    random.shuffle(danh_sach_anh)

    #tính điểm cắt
    so_luong_anh =int(len(danh_sach_anh) * SPLIT_RATIO)

    #cắt ảnh
    danh_sach_train = danh_sach_anh[:so_luong_anh]
    danh_sach_test = danh_sach_anh[so_luong_anh:]

    for ten_anh in danh_sach_train:
        duong_dan_nguon = os.path.join(raw_folder,ten_anh)
        duong_dan_dich = os.path.join(duong_dan_train_folder,ten_anh)
        shutil.copy(duong_dan_nguon, duong_dan_dich)
    print(f"Xong {loai_keo}: {len(danh_sach_train)} ảnh train.")


    for ten_anh in danh_sach_test:
        duong_dan_nguon = os.path.join(raw_folder,ten_anh)
        duong_dan_dich = os.path.join(duong_dan_test_folder,ten_anh)
        shutil.copy(duong_dan_nguon,duong_dan_dich)
    print(f"Xong {loai_keo}: {len(danh_sach_test)} ảnh test.")

