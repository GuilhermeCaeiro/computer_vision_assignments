import numpy as np
import tkinter
from PIL import Image, ImageTk
from sys import argv
import math

window = tkinter.Tk(className="my_window")

original_image = Image.open("test.png")

"""
canvas = tkinter.Canvas(window, width=original_image.size[0], height=original_image.size[1])
canvas.pack()
image_tk = ImageTk.PhotoImage(original_image)
canvas.create_image(original_image.size[0]//2, original_image.size[1]//2, image=image_tk)

def callback(event):
    print("clicked at: ", event.x, event.y)

canvas.bind("<Button-1>", callback)
tkinter.mainloop()
#"""


#"""
deformed_image_array = np.array(original_image)
deformed_image_width = deformed_image_array.shape[1]
deformed_image_height = deformed_image_array.shape[0]

print(type(deformed_image_array))
print(deformed_image_array.shape, deformed_image_width, deformed_image_height)
print(deformed_image_array[0])
print(deformed_image_array[0][0])

deformed_points = np.matrix([
	#[103, 275],
	#[108, 193],
	#[122, 199],
	#[117, 270],
	[439, 718],
	[402, 59],
	[800, 140],
	[801, 524],
])

original_points = np.matrix([
	#[0, 0],
	#[0, 600],
	#[800, 600],
	#[800, 0]
	[0, 600],
	[0, 0],
	[800, 0],
	[800, 600],

])

#original_points = original_points * 0.1

A = np.zeros((8, 8))

deformed_points_coordinates = np.asmatrix(deformed_points).reshape(-1)
original_points_coordinates = np.asmatrix(original_points).reshape(-1)

print(deformed_points_coordinates)
print(original_points_coordinates)

# generating A matrix

for i in range(0, len(deformed_points_coordinates.T)):
	point_base_position = int(2 * np.floor(i / 2))
	print(point_base_position) 
	#x = original_points_coordinates[0, point_base_position]
	#y = original_points_coordinates[0, point_base_position + 1]
	#x_line = deformed_points_coordinates[0, point_base_position]
	#y_line = deformed_points_coordinates[0, point_base_position + 1]

	x = deformed_points_coordinates[0, point_base_position]
	y = deformed_points_coordinates[0, point_base_position + 1]
	x_line = original_points_coordinates[0, point_base_position]
	y_line = original_points_coordinates[0, point_base_position + 1]

	if i % 2 == 0:
		A[i] = [x, y, 1, 0, 0, 0, - x * x_line, - y * x_line]
	else:
		A[i] = [0, 0, 0, x, y, 1, - x * y_line, - y * y_line]

print(A)

#"""
#"""
#h_values = np.linalg.solve(A, original_points_coordinates.T) # calculates x = A^-1 * b

h_values = np.linalg.inv(A) * original_points_coordinates.T

#h_values = np.linalg.inv(A.T * A) * A.T * original_points_coordinates.T # using the pseudo inverse

print(h_values)

h_values = np.append(h_values, np.matrix([[1]]), axis=0) # Adding h33

print(h_values)

homography_matrix = np.reshape(h_values, (3, 3))

homography_matrix_inverse = np.linalg.inv(homography_matrix)

print(homography_matrix)

print(homography_matrix_inverse)

#print(homography_matrix * np.matrix([[deformed_points[0, 0], deformed_points[0, 1], 1]]).T)

#print(homography_matrix_inverse * np.matrix([[original_points[0, 0], original_points[0, 1], 1]]).T)

op = np.matrix([[original_points[0, 0], original_points[0, 1], 1]])
transformed_point = homography_matrix_inverse * op.T
print(op)
print(transformed_point)
print(int(transformed_point[0,0]/transformed_point[2,0]), int(transformed_point[1,0]/transformed_point[2,0]))

print("\n")

dp = np.matrix([[deformed_points[0, 0], deformed_points[0, 1], 1]])
#print(dp, homography_matrix * dp.T)
transformed_point = homography_matrix * dp.T
print(dp)
print(transformed_point)
print(int(transformed_point[0,0]/transformed_point[2,0]), int(transformed_point[1,0]/transformed_point[2,0]))

print("\n")

#op = np.matrix([[original_points[0, 0], original_points[0, 1], 1]])
#print(original_points[0, 0], op)
#print(op, homography_matrix * op.T)

#dp = np.matrix([[deformed_points[0, 0], deformed_points[0, 1], 1]])
#print(dp, homography_matrix_inverse * dp.T)


#"""
#"""
# Defining points of interest
deformed_x_min = 0
deformed_x_max = deformed_image_width
deformed_y_min = 0
deformed_y_max = deformed_image_height

pre_corner0 = homography_matrix * np.matrix([[deformed_x_min, deformed_y_min, 1]]).T
pre_corner1 = homography_matrix * np.matrix([[deformed_x_min, deformed_y_max, 1]]).T
pre_corner2 = homography_matrix * np.matrix([[deformed_x_max, deformed_y_max, 1]]).T
pre_corner3 = homography_matrix * np.matrix([[deformed_x_max, deformed_y_min, 1]]).T

pre_corner0 = np.matrix([[int(pre_corner0[0,0] / pre_corner0[2,0]), int(pre_corner0[1,0] / pre_corner0[2,0]), int(pre_corner0[2,0] / pre_corner0[2,0])]]).T
pre_corner1 = np.matrix([[int(pre_corner1[0,0] / pre_corner1[2,0]), int(pre_corner1[1,0] / pre_corner1[2,0]), int(pre_corner1[2,0] / pre_corner1[2,0])]]).T
pre_corner2 = np.matrix([[int(pre_corner2[0,0] / pre_corner2[2,0]), int(pre_corner2[1,0] / pre_corner2[2,0]), int(pre_corner2[2,0] / pre_corner2[2,0])]]).T
pre_corner3 = np.matrix([[int(pre_corner3[0,0] / pre_corner3[2,0]), int(pre_corner3[1,0] / pre_corner3[2,0]), int(pre_corner3[2,0] / pre_corner3[2,0])]]).T

print("pre", pre_corner0, pre_corner1, pre_corner2, pre_corner3)

pre_corners = [pre_corner0, pre_corner1, pre_corner2, pre_corner3]

x_min = math.inf
y_min = math.inf
x_max = - math.inf
y_max = - math.inf

for pre_corner in pre_corners:
	if pre_corner[0, 0] < x_min:
		x_min = pre_corner[0, 0]

	if pre_corner[0, 0] > x_max:
		x_max = pre_corner[0, 0]

	if pre_corner[1, 0] < y_min:
		y_min = pre_corner[1, 0]

	if pre_corner[1, 0] > y_max:
		y_max = pre_corner[1, 0]

print("mins", x_min, y_min, x_max, y_max)

total_x = x_max - x_min
total_y = y_max - y_min

print("totals", total_x, total_y, total_y/total_x)

#print

#"""

#"""

# output
output_width = 768
output_height = int((total_y/total_x) * output_width)

#output_width = int(total_x)
#output_height = int(total_y)

print("dims", output_width, output_height)

step_x = total_x / output_width
step_y = total_y / output_height

print("step sizes", step_x, step_y)

output_image_array = np.zeros((output_height, output_width, 3), dtype=np.int)

print(type(output_image_array))
print(output_image_array.shape)

print("border points", x_min, y_min)
print(homography_matrix_inverse * np.matrix([[x_min, y_min, 1]]).T)
print("\n")

exception_counter = 0
for x in range(0, output_width):
	for y in range(0, output_height):
		#print(x_min + int(x * step_x), y_min + (y * step_y), 1)
		deformed_point = homography_matrix_inverse * np.matrix([[x_min + int(x * step_x), y_min + int(y * step_y), 1]]).T

		try:
			#print("BLA")
			#deformed_image_array[deformed_point[0, 0], deformed_point[1, 0]]
			#print(x, y, deformed_image_array[deformed_point[0, 0], deformed_point[1, 0]])
			unscaled_deformed_point = [int(deformed_point[0,0] / deformed_point[2,0]), int(deformed_point[1,0] / deformed_point[2,0])]
			#print(x, y, unscaled_deformed_point)

			if((unscaled_deformed_point[0] < 0 or unscaled_deformed_point[0] >= deformed_image_width) or (unscaled_deformed_point[1] < 0 or unscaled_deformed_point[1] >= deformed_image_height)):
				continue
			output_image_array[y, x] = deformed_image_array[unscaled_deformed_point[1], unscaled_deformed_point[0]]
			#print("ok")
			
		except:
			exception_counter = exception_counter + 1 #print("error")
	#break

print(exception_counter)

print(output_image_array[0])
print(output_image_array[0][0])
print(output_image_array.shape)

output_image = Image.fromarray(output_image_array.astype('uint8'))
output_image.show()
#output_image.save("results.bmp")


#def check_poin_inside_quadrilateral():
#	pass


#"""
"""
output_image_array = np.zeros((1000, 1000, 3), dtype=np.int)

print(type(output_image_array))
print(output_image_array.shape)

for x in range(0, 1000):
	for y in range(0, 1000):
		deformed_point = homography_matrix_inverse * np.matrix([[x, y, 1]]).T
		if x == 0 and y == 0:
			print(deformed_point)
		#if deformed_point[0, 0] >= deformed_x_min and deformed_point[0, 0] < deformed_x_max and deformed_point[1, 0] >= deformed_y_min and deformed_point[1, 0] < deformed_y_max:
		try:
			#print("BLA")
			#deformed_image_array[deformed_point[0, 0], deformed_point[1, 0]]
			unscaled_deformed_point = [int(deformed_point[0,0] / deformed_point[2,0]), int(deformed_point[1,0] / deformed_point[2,0])]
			#print(x, y, unscaled_deformed_point)
			#print(deformed_image_array[unscaled_deformed_point[0], unscaled_deformed_point[1]])
			output_image_array[x, y] = deformed_image_array[unscaled_deformed_point[0], unscaled_deformed_point[1]]
			#print("ok")
			#break
		except:
			pass#print("error")
		#break
	#break

print(output_image_array[0])
print(output_image_array[0][0])
print(output_image_array.shape)

output_image = Image.fromarray(output_image_array.astype('uint8'))
output_image.show()
#"""

"""

print(deformed_image_array.shape)
mappings_matrix = np.zeros(deformed_image_array.shape)

print(mappings_matrix.shape)
print(deformed_image_height, deformed_image_width)

for x in range(0, deformed_image_width):
	for y in range(0, deformed_image_height):
		original_point = homography_matrix * np.matrix([[x, y, 1]]).T
		unscaled_original_point = [int(original_point[0,0] / original_point[2,0]), int(original_point[1,0] / original_point[2,0]), int(original_point[2,0] / original_point[2,0])]
		mappings_matrix[y, x] = unscaled_original_point

print(mappings_matrix.shape) 


output_image_array = np.zeros((1000, 1000, 3), dtype=np.int)

#for x in range(0, 1000):
#	for y in range(0, 1000):





#"""