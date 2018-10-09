import numpy as np
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from sys import argv
import math


class ApplicationUI:
    def __init__(self):
        self.root = tkinter.Tk(className="Remove Projective Distortion")

        self.standard_font = ("Arial", "10")
        self.standard_entry_width = 10

        self.top_frame = tkinter.Frame(self.root)
        self.top_frame.pack()

        self.image_frame = tkinter.Frame(self.root)
        self.image_frame.pack()

        self.instructions_label = tkinter.Label(self.top_frame, text="Select two pairs of originally parallel lines.   ", font=self.standard_font)
        self.instructions_label.pack(side = tkinter.LEFT)

        """self.original_width_entry = tkinter.Entry(self.top_frame)
        self.original_width_entry["font"] = self.standard_font
        self.original_width_entry["width"] = self.standard_entry_width
        self.original_width_entry.pack(side = tkinter.LEFT)

        self.original_height_label = tkinter.Label(self.top_frame, text="Rectangle original height", font=self.standard_font)
        self.original_height_label["font"] = self.standard_font
        self.original_height_label.pack(side = tkinter.LEFT)

        self.original_height_entry = tkinter.Entry(self.top_frame)
        self.original_height_entry["font"] = self.standard_font
        self.original_height_entry["width"] = self.standard_entry_width
        self.original_height_entry.pack(side = tkinter.LEFT)"""

        self.select_image_button = tkinter.Button(self.top_frame, text="Select Image")
        self.select_image_button["font"] = self.standard_font
        self.select_image_button["width"] = self.standard_entry_width
        self.select_image_button["command"] = self.select_image
        self.select_image_button.pack(side = tkinter.LEFT)

        self.transform_button = tkinter.Button(self.top_frame, text="Go To Step 2")
        self.transform_button["font"] = self.standard_font
        self.transform_button["width"] = self.standard_entry_width
        self.transform_button["command"] = self.transform_image
        self.transform_button.pack(side = tkinter.LEFT)


        self.image_file_name = ""
        self.image_data = None

        self.current_step = 1
        self.clicked_points_list = []

    def run(self):
        self.root.mainloop()

    def transform_image(self):
        print("Working")

        deformed_image_array = np.array(self.image_data)
        deformed_image_width = deformed_image_array.shape[1]
        deformed_image_height = deformed_image_array.shape[0]

        #int_original_width_entry = int(self.original_width_entry.get())
        #int_original_height_entry = int(self.original_height_entry.get())

        #biggest_side_size = int_original_width_entry if int_original_width_entry > int_original_height_entry else int_original_height_entry

        # normalizing
        #rectangle_width = int((int_original_width_entry / biggest_side_size) * 1000)
        #rectangle_height = int((int_original_height_entry / biggest_side_size) * 1000)

        #print(type(deformed_image_array))
        print(deformed_image_array.shape, deformed_image_width, deformed_image_height)
        print(deformed_image_array[0])
        print(deformed_image_array[0][0])

        
        """deformed_points = np.matrix([
            [385, 308, 1],
            [404, 112, 1],
            [405, 310, 1],
            [420, 113, 1],
            [686, 287, 1],
            [651, 119, 1],
            [707, 302, 1],
            [662, 110, 1],
        ])"""

        deformed_points = np.matrix([
            [404, 114, 1],
            [386, 302, 1],
            [421, 124, 1],
            [407, 290, 1],
            [651, 119, 1],
            [686, 285, 1],
            [666, 112, 1],
            [709, 299, 1],
        ])

        """original_points = np.matrix([
            #[0, 0],
            #[0, 600],
            #[800, 600],
            #[800, 0]
            [0, 600],
            [0, 0],
            [800, 0],
            [800, 600],

        ])"""

        #deformed_points = self.clicked_points_list

        #original_points = np.matrix([
        #    [0, rectangle_height],
        #    [0, 0],
        #    [rectangle_width, 0],
        #    [rectangle_width, rectangle_height],
        #
        #])

        #original_points = original_points * 0.1

        print(deformed_points)
        line_one = np.cross(deformed_points[0], deformed_points[1])
        line_two = np.cross(deformed_points[2], deformed_points[3])
        line_three = np.cross(deformed_points[4], deformed_points[5])
        line_four = np.cross(deformed_points[6], deformed_points[7])

        print("lines", line_one, line_two, line_three, line_four)

        line_one = (line_one * (1/line_one[0, 2]))#.astype(int)
        line_two = (line_two * (1/line_two[0, 2]))#.astype(int)
        line_three = (line_three * (1/line_three[0, 2]))#.astype(int)
        line_four = (line_four * (1/line_four[0, 2]))#.astype(int)

        print("lines", line_one, line_two, line_three, line_four)

        point_one_in_the_infinite = np.cross(line_one, line_two)
        point_two_in_the_infinite = np.cross(line_three, line_four)

        print("points in the infinite", point_one_in_the_infinite, point_two_in_the_infinite)

        line_in_the_infinite = np.cross(point_one_in_the_infinite, point_two_in_the_infinite)

        print("line in the infinite", line_in_the_infinite)

        # matrix to transform from projective space to affine
        projective_homography_matrix = np.matrix([
            [1, 0, 0], 
            [0, 1, 0], 
            [line_in_the_infinite[0, 0], line_in_the_infinite[0, 1], line_in_the_infinite[0, 2]]
        ])

        print(projective_homography_matrix)


        affine_homography_matrix = None






























        """

        A = np.zeros((8, 8))

        deformed_points_coordinates = np.asmatrix(deformed_points).reshape(-1)
        original_points_coordinates = np.asmatrix(original_points).reshape(-1)

        print(deformed_points_coordinates)
        print(original_points_coordinates)

        # generating A matrix

        for i in range(0, len(deformed_points_coordinates.T)):
            point_base_position = int(2 * np.floor(i / 2))
            #print(point_base_position) 
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

        #h_values = np.linalg.solve(A, original_points_coordinates.T) # calculates x = A^-1 * b

        h_values = np.linalg.inv(A) * original_points_coordinates.T

        #h_values = np.linalg.inv(A.T * A) * A.T * original_points_coordinates.T # using the pseudo inverse

        #print(h_values)

        h_values = np.append(h_values, np.matrix([[1]]), axis=0) # Adding h33

        #print(h_values)

        homography_matrix = np.reshape(h_values, (3, 3))

        homography_matrix_inverse = np.linalg.inv(homography_matrix)

        #print(homography_matrix)

        #print(homography_matrix_inverse)


        #"""

        """op = np.matrix([[original_points[0, 0], original_points[0, 1], 1]])
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

        #"""

        #homography_matrix = affine_homography_matrix * projective_homography_matrix
        
        #homography_matrix = np.linalg.inv(projective_homography_matrix)
        #homography_matrix_inverse = projective_homography_matrix
        homography_matrix = projective_homography_matrix
        homography_matrix_inverse = np.linalg.inv(projective_homography_matrix)

        # Defining points of interest
        deformed_x_min = 0
        deformed_x_max = deformed_image_width
        deformed_y_min = 0
        deformed_y_max = deformed_image_height

        pre_corner0 = homography_matrix * np.matrix([[deformed_x_min, deformed_y_min, 1]]).T
        pre_corner1 = homography_matrix * np.matrix([[deformed_x_min, deformed_y_max, 1]]).T
        pre_corner2 = homography_matrix * np.matrix([[deformed_x_max, deformed_y_max, 1]]).T
        pre_corner3 = homography_matrix * np.matrix([[deformed_x_max, deformed_y_min, 1]]).T

        print("pre", pre_corner0, pre_corner1, pre_corner2, pre_corner3)

        pre_corner0 = np.matrix([[int(pre_corner0[0,0] / pre_corner0[2,0]), int(pre_corner0[1,0] / pre_corner0[2,0]), int(pre_corner0[2,0] / pre_corner0[2,0])]]).T
        pre_corner1 = np.matrix([[int(pre_corner1[0,0] / pre_corner1[2,0]), int(pre_corner1[1,0] / pre_corner1[2,0]), int(pre_corner1[2,0] / pre_corner1[2,0])]]).T
        pre_corner2 = np.matrix([[int(pre_corner2[0,0] / pre_corner2[2,0]), int(pre_corner2[1,0] / pre_corner2[2,0]), int(pre_corner2[2,0] / pre_corner2[2,0])]]).T
        pre_corner3 = np.matrix([[int(pre_corner3[0,0] / pre_corner3[2,0]), int(pre_corner3[1,0] / pre_corner3[2,0]), int(pre_corner3[2,0] / pre_corner3[2,0])]]).T

        #pre_corner0 = pre_corner0.astype(int)
        #pre_corner1 = pre_corner1.astype(int)
        #pre_corner2 = pre_corner2.astype(int)
        #pre_corner3 = pre_corner3.astype(int)

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

        # output
        output_width = 768
        output_height = int((total_y/total_x) * output_width)

        #output_width = int(total_x)
        #output_height = int(total_y)

        #print("dims", output_width, output_height)

        step_x = total_x / output_width
        step_y = total_y / output_height

        #print("step sizes", step_x, step_y)

        output_image_array = np.zeros((output_height, output_width, 3), dtype=np.int)

        #print(type(output_image_array))
        print(output_image_array.shape)

        print("border points", x_min, y_min)
        print(homography_matrix_inverse * np.matrix([[x_min, y_min, 1]]).T)
        #print("\n")

        for x in range(0, output_width):
            for y in range(0, output_height):
                deformed_point = homography_matrix_inverse * np.matrix([[x_min + int(x * step_x), y_min + int(y * step_y), 1]]).T

                try:
                    unscaled_deformed_point = [int(deformed_point[0,0] / deformed_point[2,0]), int(deformed_point[1,0] / deformed_point[2,0])]

                    if((unscaled_deformed_point[0] < 0 or unscaled_deformed_point[0] >= deformed_image_width) or (unscaled_deformed_point[1] < 0 or unscaled_deformed_point[1] >= deformed_image_height)):
                        continue

                    output_image_array[y, x] = deformed_image_array[unscaled_deformed_point[1], unscaled_deformed_point[0]]
                    
                except:
                    pass#print("error")


        #print(output_image_array[0])
        #print(output_image_array[0][0])
        print(output_image_array.shape)

        output_image = Image.fromarray(output_image_array.astype('uint8'))
        #output_image.show()

        self.draw_result(output_image)

    def select_image(self):
        self.image_file_name = filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("jpeg files","*.jpg"), ("all files","*.*"))) 
        self.image_data = Image.open(self.image_file_name)

        self.image_canvas = tkinter.Canvas(self.image_frame, width=self.image_data.size[0], height=self.image_data.size[1])
        self.image_canvas.pack()
        
        image_tk = ImageTk.PhotoImage(self.image_data)
        self.image_canvas.create_image(self.image_data.size[0]//2, self.image_data.size[1]//2, image=image_tk)

        self.image_canvas.bind("<Button-1>", self.get_click_position)

        # updating
        self.run()

    def get_click_position(self, event):
        print("clicked at: ", event.x, event.y)

        x = event.x
        y = event.y
        r = 10 # radius

        # TODO Change the else condition value
        maximun_points = 8 if self.current_step == 1 else 0

        if len(self.clicked_points_list) < maximun_points:
            self.image_canvas.create_oval(x-r, y-r, x+r, y+r, width=3, outline="#fb0")

            self.clicked_points_list.append([x, y, 1])

        else:
            self.image_canvas.create_oval(x-r, y-r, x+r, y+r, width=3, outline="red")

    def calculate_homography(self):
        pass

    def draw_result(self, output_image):       
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        self.image_data = output_image

        self.image_canvas = tkinter.Canvas(self.image_frame, width=self.image_data.size[0], height=self.image_data.size[1])
        self.image_canvas.pack()
        
        image_tk = ImageTk.PhotoImage(self.image_data)
        self.image_canvas.create_image(self.image_data.size[0]//2, self.image_data.size[1]//2, image=image_tk)

        self.image_canvas.bind("<Button-1>", self.get_click_position)

        self.run()

application = ApplicationUI()
application.run()