#GUI/dicom.py
import pydicom
import os

# DICOMファイルを読み込む
ex_root = './DATA/EXFILES/EX7'

def find_series(EX_Dir,series): #->EX_dir内の目的のSE_Dir
    for SE in os.listdir(EX_Dir):
        SE_Dir=os.path.join(EX_Dir, SE)
        dicom_file=os.path.join(SE_Dir,"IMG1")
        try:
            ds = pydicom.dcmread(dicom_file)
            SeriesDescription= ds.SeriesDescription
            if SeriesDescription == series:
                return SE_Dir
        except:
                continue

se_dir = find_series(ex_root,"eT1W_SE_cor")
print(se_dir)
if se_dir is not None:
    for dicom_file in os.listdir(se_dir):
        file = os.path.join(se_dir, dicom_file)
        ds = pydicom.dcmread(file)
        # メタデータを表示する
        # print(ds)
        # 特定のタグ番号を使って情報を取り出す場合 (例：0008,0060 はモダリティ)
        # modality = ds[0x0008, 0x0060].value
        # print(f"Modality: {modality}")

        SeriesDescription= ds.SeriesDescription
        print(SeriesDescription)
else:
     print("NOne")