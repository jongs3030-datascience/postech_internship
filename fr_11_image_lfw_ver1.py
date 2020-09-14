# for the LFW pair condition
import insightface
import cv2
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
import re

def input_arrange(line):
	line = line.split('\t')
	line1 = line[0]
	line2 = line[1]


	person1 = line1.split('/')[0]
	person2 = line2.split('/')[0]
	return person1, person2, line1, line2


def image_crop(img, bbox):
	f_box = [69, 69, 181, 181]
	bbox = bbox.astype(np.int).flatten()
	print("bbox : ", bbox)
	b_idx = 0
	# expand_ratio = 0.1
	expand_ratio = 0.1
	while b_idx < len(bbox):
		if bbox[b_idx]<125 and bbox[b_idx+1]<125 and bbox[b_idx+2]>125 and bbox[b_idx+3]>125 :
			f_box[0] = int((1-expand_ratio)*bbox[b_idx])
			f_box[1] = int((1-expand_ratio)*bbox[b_idx+1])
			f_box[2] = int((1+expand_ratio)*bbox[b_idx+2])
			f_box[3] = int((1+expand_ratio)*bbox[b_idx+3])
			face_img = img[f_box[1]:f_box[3], f_box[0]:f_box[2]]
			break
		else:
			b_idx += 5
	print("f_box : ", f_box)
	if f_box == [69, 69, 181, 181]:
		print('Face Detection Fail!')
		face_img = img[f_box[1]:f_box[3], f_box[0]:f_box[2]]
	return face_img


def image_rotate(img, landmark):
	angle = np.arctan((landmark[0][1][1] - landmark[0][0][1]) / (landmark[0][1][0] - landmark[0][0][0])) * 180 / np.pi
	M1 = cv2.getRotationMatrix2D((112/2, 112/2), angle, 1)
	face_img = cv2.warpAffine(img, M1, (112, 112))
	return face_img


def face_draw(bbox, landmark, img):
	for box, land in zip(bbox, landmark):
		cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
		for (x, y) in land:
			cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)



# Model Loading : Deteciton and Recognition
face_detect_model = insightface.model_zoo.get_model('retinaface_r50_v1')
face_detect_model.prepare(ctx_id = 0, nms=0.4)

face_recognition_model = insightface.model_zoo.get_model('arcface_r100_v1')
face_recognition_model.prepare(ctx_id = 0)

# file path
# file_path = '/home/pirl/PycharmProjects/internship_project/'
file_path = '/home/pirl/PycharmProjects/test_folder/'

# Input Parameters
son = ''
idx = 0
input_file = 'pairs_lfw.txt'   # LFW pair file location
# input_file = 'entertainer_pairs.txt'   # LFW pair file location
i_f = open(input_file)

# output_file = 'output_11_image_lfw_test.csv'   # output csv file name
# output_file = 'heuristic_face_rotate_output.csv'   # output csv file name
# output_file = 'original_resized_heuristic.csv'   # output csv file name
# output_file = 'fill_extra_image.csv'   # output csv file name
# output_file = 'entertainer_test.csv'   # output csv file name
output_file = 'expanded(0)_crop_vs_heuristic_vs_crop_heuristic_vs_extra.csv'   # output csv file name

o_f = open(output_file, 'w', encoding='utf-8', newline='')
wr = csv.writer(o_f)

iter = 1
total_length = []
total_height = []

############################저장을 위한 파라미터#####################################
red = (20, 20, 255)
green = (20,255,20)
center_x = int(80)
center_y = int(20)
location = (center_x, center_y)
location2 = (0, center_y)

# filled_location = (int(80), int(130))
# filled_location2 = (0, int(130))
font = cv2.FONT_HERSHEY_COMPLEX  # normal size serif font
fontScale = 0.5
thickness = 1

img0 = np.zeros((30, 224 , 3), np.uint8)
##############################################################

# Main Running
while True:
	line = i_f.readline()
	if not line:
		break
	idx += 1

	line1, line2, label = line.split()[:3]
	# print("line : ", line)
	# print("line1 : ", line1)
	# print("line2 : ", line2)
	# print("label : ", label)
	# Image reading
	# print("input_arrange(line) : ", input_arrange(line))
	person1, person2, line1, line2 = input_arrange(line)

	# print("person1 : ", person1)
	# print("person1 : ", person2)
	# print("line1 : ", line1)
	# print("line2 : ", line2)
	# break
	son = 1 if person1==person2 else 0   # son : same or not, same = 1, different = 0

############################# dataset setting #################################
	img_path1 = 'dataset_image/{}'.format(line1)   # image file path format
	img_path2 = 'dataset_image/{}'.format(line2)

	# img_path1 = 'test_img/{}'.format(line1)   # image file path format
	# img_path2 = 'test_img/{}'.format(line2)
#############################  #################################

	img1 = cv2.imread(img_path1)
	img2 = cv2.imread(img_path2)

	# Face detection
	bbox1, landmark1 = face_detect_model.detect(img1, threshold=0.5, scale=1.0)
	bbox2, landmark2 = face_detect_model.detect(img2, threshold=0.5, scale=1.0)

	# print(bbox1)
	# print(bbox2)

	# Face Cropping
	face_img1 = image_crop(img1, bbox1)
	face_img2 = image_crop(img2, bbox2)

	bbox1 = bbox1.astype(np.int).flatten()
	raw_image = img1[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]

	fig = plt.figure()
	rows = 1
	cols = 2

	ax1 = fig.add_subplot(rows, cols, 1)
	ax1.imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
	ax1.set_title('raw image 1')
	ax1.axis("off")

	ax2 = fig.add_subplot(rows, cols, 2)
	ax2.imshow(cv2.cvtColor(face_img1, cv2.COLOR_BGR2RGB))
	ax2.set_title('cropped image 2')
	ax2.axis("off")

	plt.show()
	break

################ 가로 사이즈 만큼 세로 사이즈 자를건데 가운데 기준 +-로 #################

	height1, width1, channels1 = face_img1.shape
	height2, width2, channels2 = face_img2.shape

	margin1 = int(width1/2-10)
	margin2 = int(width2/2-10)

	mid_h1 = int(height1/2)
	mid_h2 = int(height2/2)

	mid_w1 = int(width1/2)
	mid_w2 = int(width2/2)

####### full face heuristic ##########################
	# heu_face_img1 = face_img1[mid_h1 - margin1:mid_h1 + margin1, mid_w1 - int(margin1):mid_w1 + int(margin1)]
	# heu_face_img2 = face_img2[mid_h2 - margin2:mid_h2 + margin2, mid_w2 - int(margin2):mid_w2 + int(margin2)]

####### width 사이즈 유지, height 길이는 중심점 기준으로 width랑 동일하게 ###############
	crop_heu_face_img1 = face_img1[mid_h1 - mid_w1:mid_h1 + mid_w1, 0:width1]
	crop_heu_face_img2 = face_img2[mid_h2 - mid_w2:mid_h2 + mid_w2, 0:width2]


	# (Optional) Heuristic Face Detection
	heu_face_img1 = img1[60:190, 60:190]
	heu_face_img2 = img2[60:190, 60:190]

############################################
	# plt.imshow(face_img1)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


	# fig = plt.figure()
	# rows = 1
	# cols = 2
	#
	# ax1 = fig.add_subplot(rows, cols, 1)
	# ax1.imshow(cv2.cvtColor(face_img1, cv2.COLOR_BGR2RGB))
	# ax1.set_title('image 1')
	# ax1.axis("off")
	#
	# ax2 = fig.add_subplot(rows, cols, 2)
	# ax2.imshow(cv2.cvtColor(heu_face_img1, cv2.COLOR_BGR2RGB))
	# ax2.set_title('image 2')
	# ax2.axis("off")
	#
	# plt.show()
	# break

	# 평균 가로 길이, 세로 길이 계산하기
	# total_height.append(face_img1.shape[0])
	# total_height.append(face_img2.shape[0])
	# total_length.append(face_img1.shape[1])
	# total_length.append(face_img2.shape[1])

#################### 여백 추가한 nxn 이미지로 만들어 유사도를 비교해보자 ####################

	height1, width1, channels1 = face_img1.shape
	height2, width2, channels2 = face_img2.shape

	height_width_difference1 = int((face_img1.shape[0] - face_img1.shape[1])/2)
	height_width_difference2 = int((face_img2.shape[0] - face_img2.shape[1])/2)

	#print("face_img shape : ", face_img1.shape)
	blank_image_1 = np.zeros((height1, height1, 3), np.uint8)
	blank_image_2 = np.zeros((height2, height2, 3), np.uint8)

	#print("blank_img shape : ", blank_image.shape)

	blank_image_1[0 : height1, height_width_difference1 : height_width_difference1 + width1] = face_img1
	blank_image_2[0 : height2, height_width_difference2 : height_width_difference2 + width2] = face_img2

	# fig = plt.figure()
	# rows = 2
	# cols = 2
	#
	# ax1 = fig.add_subplot(rows, cols, 1)
	# ax1.imshow(cv2.cvtColor(blank_image_1, cv2.COLOR_BGR2RGB))
	# ax1.set_title('blank image 1')
	# ax1.axis("off")
	#
	# ax2 = fig.add_subplot(rows, cols, 2)
	# ax2.imshow(cv2.cvtColor(blank_image_2, cv2.COLOR_BGR2RGB))
	# ax2.set_title('blank image 2')
	# ax2.axis("off")
	# plt.show()


#######################################################################################

############################################

	# Image Resizing
	# test용으로 resized 안하고 모델 돌리
	# print(face_img1.shape, face_img2.shape)



	face_img1 = cv2.resize(face_img1, (112, 112))
	face_img2 = cv2.resize(face_img2, (112, 112))

	extra_face_img1 = cv2.resize(blank_image_1, (112, 112))
	extra_face_img2 = cv2.resize(blank_image_2, (112, 112))
############# 여백에 얼굴 추가한 이미지로 resize 하기
	# face_img1 = cv2.resize(blank_image_1, (112, 112))
	# face_img2 = cv2.resize(blank_image_2, (112, 112))
############################################

	crop_heu_face_img1 = cv2.resize(crop_heu_face_img1, (112, 112))
	crop_heu_face_img2 = cv2.resize(crop_heu_face_img2, (112, 112))

############# heuristic 얼굴 resize ############################

	heu_face_img1 = cv2.resize(heu_face_img1, (112, 112))
	heu_face_img2 = cv2.resize(heu_face_img2, (112, 112))

###############################################################

	# plt.imshow(face_img1)
	# plt.show()
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	#plt.show()


	# ax3 = fig.add_subplot(rows, cols, 3)
	# ax3.imshow(cv2.cvtColor(face_img1, cv2.COLOR_BGR2RGB))
	# ax3.set_title('resized image 1')
	# ax3.axis("off")
	#
	# ax4 = fig.add_subplot(rows, cols, 4)
	# ax4.imshow(cv2.cvtColor(face_img2, cv2.COLOR_BGR2RGB))
	# ax4.set_title('resized image 2')
	# ax4.axis("off")
	#
	# plt.show()


	# iter += 1
	# if iter == 3:
	# 	break
############################################

	# (Optional) Face Alignment(Image Rotate)
	# face_img1 = image_rotate(face_img1, landmark1)
	# face_img2 = image_rotate(face_img2, landmark2)

	# Face recognition - Similarity Result
	sim = face_recognition_model.compute_sim(face_img1, face_img2)
	# filled_sim = face_recognition_model.compute_sim(filled_face_img1, filled_face_img2)
	heu_sim = face_recognition_model.compute_sim(heu_face_img1, heu_face_img2)
	crop_heu_sim = face_recognition_model.compute_sim(crop_heu_face_img1, crop_heu_face_img2)
	extra_sim = face_recognition_model.compute_sim(extra_face_img1, extra_face_img2)

	sim = round(sim, 3)
	heu_sim = round(heu_sim, 3)
	crop_heu_sim = round(crop_heu_sim, 3)
	extra_sim = round(extra_sim, 3)

	img_name_idx1 = re.search(r"/.*\.", line1).group()[1:-1]
	img_name_idx2 = re.search(r"/.*\.", line2).group()[1:-1]

	# (Optional) Object Detection Result Checking
	face_img = cv2.hconcat([face_img1, face_img2])
	face_img = cv2.vconcat([img0, face_img])

	# 텍스트 넣기
	if float(sim) > 0:
		color1 = green
	else:
		color1 = red

	if label == '1':
		color2 = green
		label_text = 'same'
	else:
		color2 = red
		label_text = 'differ'

	# print(sim, location, font, fontScale, color1)
	# print(label, label_text, location2, font, fontScale, color2)


	# fig = plt.figure()
	# rows = 1
	# cols = 1
	#
	# ax1 = fig.add_subplot(rows, cols, 1)
	# ax1.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
	# ax1.set_title('blank image 1')
	# ax1.axis("off")
	# plt.show()


	# 텍스트 공란에 넣기
	cv2.putText(face_img, str(sim) + " (crop)", location, font, fontScale, color1)
	cv2.putText(face_img, label_text, location2, font, fontScale, color2)

	# filled_img = cv2.hconcat([filled_face_img1, filled_face_img2])
	# filled_img = cv2.vconcat([img0, filled_img])

	heu_img = cv2.hconcat([heu_face_img1, heu_face_img2])
	heu_img = cv2.vconcat([img0, heu_img])

	# 텍스트 넣기
	if float(heu_sim) > 0:
		color1 = green
	else:
		color1 = red

	if label == '1':
		color2 = green
		label_text = 'same'
	else:
		color2 = red
		label_text = 'differ'

	cv2.putText(heu_img, str(heu_sim) + " (heu)", location, font, fontScale, color1)
	cv2.putText(heu_img, label_text, location2, font, fontScale, color2)

######### crop_heuristic_face 이미지 ######################

	crop_heu_img = cv2.hconcat([crop_heu_face_img1, crop_heu_face_img2])
	crop_heu_img = cv2.vconcat([img0, crop_heu_img])

	# 텍스트 넣기
	if float(crop_heu_sim) > 0:
		color1 = green
	else:
		color1 = red

	if label == '1':
		color2 = green
		label_text = 'same'
	else:
		color2 = red
		label_text = 'differ'

	cv2.putText(crop_heu_img, str(crop_heu_sim)+" (crop_heu)", location, font, fontScale, color1)
	cv2.putText(crop_heu_img, label_text, location2, font, fontScale, color2)

##############################################################

######### extra_face 이미지 ######################

	extra_img = cv2.hconcat([extra_face_img1, extra_face_img2])
	extra_img = cv2.vconcat([img0, extra_img])

	# 텍스트 넣기
	if float(extra_sim) > 0:
		color1 = green
	else:
		color1 = red

	if label == '1':
		color2 = green
		label_text = 'same'
	else:
		color2 = red
		label_text = 'differ'

	cv2.putText(extra_img, str(extra_sim) + " (extra)", location, font, fontScale, color1)
	cv2.putText(extra_img, label_text, location2, font, fontScale, color2)

	##############################################################

	# four_img = cv2.vconcat([face_img, heu_img])
	eight_img = cv2.vconcat([face_img, heu_img, crop_heu_img, extra_img])

	# fig = plt.figure()
	# rows = 1
	# cols = 1
	#
	# ax1 = fig.add_subplot(rows, cols, 1)
	# ax1.imshow(cv2.cvtColor(eight_img, cv2.COLOR_BGR2RGB))
	# ax1.set_title('blank image 1')
	# ax1.axis("off")
	# plt.show()
	# break




	# cv2.imwrite('/home/pirl/PycharmProjects/test_folder/concat_result_image/{}_{}_{}.jpg'.format(idx, person1, person2), four_img)
	cv2.imwrite(f'expanded(0)_crop_vs_heuristic_vs_crop_heuristic_vs_extra/{idx}_{person1}_VS_{person2}.jpg', eight_img)
	# cv2.imwrite(f'entertainer_test/{idx}_{person1}_VS_{person2}.jpg', eight_img)




	# Output
	wr.writerow([idx, son, sim, heu_sim, crop_heu_sim, extra_sim])
	if idx % 100 == 0 :

		print('{:2.2%} complete!'.format(idx/6000))


i_f.close()
o_f.close()