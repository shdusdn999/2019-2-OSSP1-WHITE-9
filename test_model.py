import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model
from imageai.Detection import ObjectDetection
import os
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'


def get_args(): #콘솔로부터 인자를 받아오는 함수
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):   #이미지를 받아오는 함수
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main(): 
    # get_args() 함수로부터 인자를 받아옴
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    val_noise_model = get_noise_model(args.test_noise_model)
    model = get_model(args.model)
    model.load_weights(weight_file)

    if args.output_dir: #out_path를 입력받았다면
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:  #image_path 내에 있는 모든 이미지 파일을 순차적으로 작업함
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape
        image = image[:(h // 16) * 16, :(w // 16) * 16] 
        h, w, _ = image.shape
        
        out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
        noise_image = val_noise_model(image)
        pred = model.predict(np.expand_dims(noise_image, 0))
        denoised_image = get_image(pred[0])
 
        #out_image에 imgae, noise_image, denoised_image를 넣음.(사용하지 않음)
        execution_path = os.getcwd()
        out_image[:, :w] = image
        out_image[:, w:w * 2] = noise_image
        out_image[:, w * 2:] = denoised_image
        #cv2.imwrite("denoised.jpg", out_image)
        cv2.imwrite("noised.jpg", noise_image)
        cv2.imwrite("denoised.jpg", denoised_image) #denoised된 이미지를 기본경로에 저장
        #경로 내에 존재하는 denoised.jpg파일을 tesseract로 불러옴
        #data = pytesseract.image_to_string(Image.open('denoised.jpg'), lang='eng')
        #불러온 image 내의 문자열을 data 변수에 string으로 입력
        #f = open("result.txt", 'w', encoding='UTF8')
        #f.write(data)   #result.txt파일 안에 해당 string을 저장하고 기본 경로에 저장.
        #f.close()

        #img = cv2.imread(execution_path, "denoised.jpg")
        
        if args.output_dir: #output_path를 입력받았다면 그 경로에 저장
            cv2.imwrite("denoised.png", out_image)
        else:
            #cv2.imshow("result", out_image)

            detector = ObjectDetection()    #Object Detection을 하기 위해 ObjectDetection()호출
            detector.setModelTypeAsRetinaNet() 
            detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
            detector.loadModel()
            #denoised이미지를 불러와 object detection 시킨 이미지를 detectedImage.jpg로 저장함
            detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "denoised.jpg"), output_image_path=os.path.join(execution_path , "detectedImage.jpg"))

            for eachObject in detections:
                # 콘솔 창에도 인식된 물체의 이름과 정확도를 출력시킴
                print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
            
            #detectedImage를 imread로 새로운 창에 띄우기 위한 과정
            img = cv2.imread('detectedImage.jpg', cv2.IMREAD_UNCHANGED)
            #resized_img = cv2.resize(img, (680, 680))
            #result라는 이름의 윈도우 창으로 화면에 띄움
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("result", 680,680)
            cv2.imshow('result', img)
            
            print ("Press q if your works are done")
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0


if __name__ == '__main__':
    main()
