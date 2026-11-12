import cv2
import os
import re

def cat_frame_tu_video(duong_dan_video, thu_muc_luu, ten_video_goc, buoc_nhay):
    if not os.path.exists(thu_muc_luu):
        os.makedirs(thu_muc_luu)

    #Mở đường dẫn video
    cap=cv2.VideoCapture(duong_dan_video)
    if not cap.isOpened():
        print(f"Lỗi: Không mở được video: {duong_dan_video}")
        return

    dem_frame = 0 #đếm frame đã chạy
    dem_anh_luu = 0 #đếm ảnh đã lưu xuống

    while True:
        ret,frame=cap.read()

        if not ret:
            break

        if dem_frame % buoc_nhay == 0:
            #resize theo chuẩn đề bài
            anh_chuan_hoa = cv2.resize(frame, (256,256))

            #đặt tên file
            ten_file = os.path.join(thu_muc_luu,f"{ten_video_goc}_{dem_anh_luu:04d}.jpg")

            #lưu ảnh xuống ổ cứng
            cv2.imwrite(ten_file,anh_chuan_hoa)
            dem_anh_luu +=1

        dem_frame +=1

    #đóng và lưu kết quả
    cap.release()
    print(f"Đã cắt xong~ Tổng cộng được {dem_anh_luu} bức ảnh vào{thu_muc_luu}")
def xu_ly_tat_ca_video(thu_muc_video,thu_muc_dataset,buoc_nhay):
    #định dạng video hỗ trợ 
    dinh_dang_hop_le=('.mp4','.avi','.mov','.mkv')

    danh_sach_file=os.listdir(thu_muc_video)

    video_count=0
    for ten_file in danh_sach_file:
        #check định dạng hợp lệ
        if(ten_file.lower().endswith(dinh_dang_hop_le)):
            video_count+=1
            #tách lấy tên kẹo không có đuôi
            ten_keo_co_duoi=os.path.splitext(ten_file)[0]

            ten_keo=re.sub(r'_\d+$', '', ten_keo_co_duoi)

            #đường dẫn video
            duong_dan_video=os.path.join(thu_muc_video,ten_file)

            #tạo thư mục con tương ứng với tên kẹo 
            thu_muc_luu=os.path.join(thu_muc_dataset,ten_keo)

            print(f"[{video_count}] Đang xử lý: {ten_file} ...")
            cat_frame_tu_video(duong_dan_video,thu_muc_luu,ten_keo_co_duoi,buoc_nhay)

    if(video_count==0):
        print(f"[!] không tìm thấy video nào trong trung mục raw_videos")
    else:
        print(f"[!]Thành công! Đã xử lý xong toàn bộ {video_count} video")


if __name__ == "__main__":
    THU_MUC_CHUA_VIDEO = "data/raw_videos"
    THU_MUC_LUU_ANH = "data/raw_data_set"
    xu_ly_tat_ca_video(THU_MUC_CHUA_VIDEO,THU_MUC_LUU_ANH,15)


        


