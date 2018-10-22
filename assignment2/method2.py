import numpy as np
import scipy.linalg
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
        self.transform_button["command"] = self.transform_to_similarity
        self.transform_button.pack(side = tkinter.LEFT)


        self.image_file_name = ""
        self.image_data = None

        self.current_step = 1
        self.clicked_points_list = []

    def run(self):
        self.root.mainloop()

    def transform_to_similarity(self):
        print("Working")


        deformed_points = np.matrix([
            [493, 368, 1],
            [6, 35, 1],
            [41, 155, 1],
            [291, 24, 1],
            
            [493, 368, 1],
            [6, 35, 1],
            [29, 375, 1],
            [460, 54, 1],
            
            [107, 28, 1],
            [495, 246, 1],
            [41, 155, 1],
            [291, 24, 1],

            [107, 28, 1],
            [495, 246, 1],
            [264, 352, 1],
            [471, 154, 1],
            
            [192, 75, 1],
            [164, 274, 1],
            [92, 211, 1],
            [418, 204, 1],
        ])


        # TODO Uncommet
        #deformed_points = self.clicked_points_list

        """print(deformed_points)
        line_one = np.cross(deformed_points[0], deformed_points[1])
        line_two = np.cross(deformed_points[2], deformed_points[3])
        line_three = np.cross(deformed_points[4], deformed_points[5])
        line_four = np.cross(deformed_points[6], deformed_points[7])

        print("lines", line_one, line_two, line_three, line_four)

        line_one = (line_one * (1/line_one[0, 2]))#.astype(int)
        line_two = (line_two * (1/line_two[0, 2]))#.astype(int)
        line_three = (line_three * (1/line_three[0, 2]))#.astype(int)
        line_four = (line_four * (1/line_four[0, 2]))#.astype(int)

        print("lines", line_one, line_two, line_three, line_four)"""

        lines = []

        for i in range(0, 20, 2):
            point_1 = deformed_points[i]
            point_2 = deformed_points[i + 1]

            line = np.cross(point_1, point_2)

            line = (line * (1 / line[0, 2]))

            lines.append(line)

        print(lines)

        system_of_equations = []
        #b = []

        for i in range(0, 10, 2):
            l = lines[i]
            m = lines[i + 1]

            system_of_equations.append([
                (l[0, 0] * m[0, 0]), 
                ((l[0, 0] * m[0, 1]) + (l[0, 1] * m[0, 0])) / 2,
                (l[0, 1] * m[0, 1]),
                ((l[0, 0] * m[0, 2]) + (l[0, 2] * m[0, 0])) / 2,
                ((l[0, 1] * m[0, 2]) + (l[0, 2] * m[0, 1])) / 2,
                (l[0, 2] * m[0, 2]),
            ])

            #b.append((l[0, 2] * m[0, 2]))

        system_of_equations.append([0, 0, 0, 0, 0, 1]) # last number is 1 to force f = 1

        print(system_of_equations)

        b = np.matrix([[0, 0, 0, 0, 0, 1]]) # last number is 1 to force f = 1
        #b = np.matrix([b])

        x = np.linalg.inv(system_of_equations) * b.T

        print("x: ", x)

        conic = np.matrix([
            [x[0, 0], x[1, 0] / 2, x[3, 0] / 2],
            [x[1, 0] / 2, x[2, 0], x[4, 0] / 2],
            [x[3, 0] / 2, x[4, 0] / 2, 1.0]
        ])

        projective_homography = None

        print(projective_homography)

        affine_homography = None

        print(affine_homography)

        # matrix to transform from projective space to similarity ()
        projective_to_similarity_homography_matrix = affine_homography * projective_homography

        print(projective_to_similarity_homography_matrix)

        output_image_array = self.transform(projective_to_similarity_homography_matrix, np.array(self.image_data))
        self.image_data = output_image_array

        output_image = Image.fromarray(output_image_array.astype('uint8'))
        output_image.save("projective_to_affine.bmp")
        #output_image.show()
        print("Drawing")
        self.draw_result(output_image)





    def transform(self, homography, deformed_image_array):
        homography_matrix = homography
        homography_matrix_inverse = np.linalg.inv(homography_matrix)

        deformed_image_width = deformed_image_array.shape[1]
        deformed_image_height = deformed_image_array.shape[0]

        print(deformed_image_array.shape, deformed_image_width, deformed_image_height)
        print(deformed_image_array[0])
        print(deformed_image_array[0][0])

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

        return output_image_array




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

        if self.current_step == 1:
            maximun_points = 8
        elif self.current_step == 2:
            maximun_points = 6
        else:
            maximun_points = 0

        if len(self.clicked_points_list) < maximun_points:
            self.image_canvas.create_oval(x-r, y-r, x+r, y+r, width=3, outline="#fb0")

            self.clicked_points_list.append([x, y, 1])

        else:
            self.image_canvas.create_oval(x-r, y-r, x+r, y+r, width=3, outline="red")

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