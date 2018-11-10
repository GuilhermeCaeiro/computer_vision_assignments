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
        self.transform_button["command"] = self.transform_to_affine
        self.transform_button.pack(side = tkinter.LEFT)


        self.image_file_name = ""
        self.image_data = None

        self.current_step = 1
        self.clicked_points_list = []

    def run(self):
        self.root.mainloop()

    def transform_to_affine(self):
        print("Working")

        

        
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

        """deformed_points = np.matrix([
            [244, 343, 1],
            [169, 297, 1],
            [269, 307, 1],
            [213, 263, 1],
            [264, 352, 1],
            [329, 291, 1],
            [230, 323, 1],
            [315, 250, 1],
        ])


        deformed_points = np.matrix([
            [172, 282, 1],
            [169, 279, 1],
            [215, 265, 1],
            [212, 262, 1],
            [231, 226, 1],
            [239, 220, 1],
            [269, 241, 1],
            [274, 237, 1],
        ])"""

        deformed_points = np.matrix([
            [17, 385, 1],
            [334, 149, 1],
            [228, 385, 1],
            [472, 153, 1],
            [280, 374, 1],
            [38, 163, 1],
            [506, 377, 1],
            [34, 54, 1],
        ])


        # TODO Uncommet
        #deformed_points = self.clicked_points_list

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


        #affine_homography_matrix = None


        output_image_array = self.transform(projective_homography_matrix, np.array(self.image_data))

        #output_image_array = output_image_array.T

        self.image_data = output_image_array#.T

        output_image = Image.fromarray(output_image_array.astype('uint8'))
        output_image.save("projective_to_affine.bmp")
        #output_image.show()
        print("Drawing")
        self.draw_result(output_image, self.uptade_interface_to_step_two)


    def transform_to_similarity(self):
        print("Transform to Similarity")

        #deformed_points = np.matrix(self.clicked_points_list)

        deformed_points = np.matrix([
            [231, 180, 1],
            [280, 213, 1],
            [280, 213, 1],
            [380, 179, 1],
            [578, 11, 1],
            [479, 45, 1],
            [479, 45, 1],
            [530, 77, 1],
        ])

        deformed_points = np.matrix([
            [288, 210, 1],
            [625, 95, 1],
            [356, 66, 1],
            [557, 192, 1],
            #[431, 79, 1],
            #[632, 76, 1],
            #[632, 76, 1],
            #[581, 144, 1],

            [584, 13, 1],
            [719, 98, 1],
            [643, 39, 1],
            [338, 143, 1],
        ])

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

        A = np.matrix([
            [line_one[0, 0] * line_two[0, 0], line_one[0, 0] * line_two[0, 1] + line_one[0, 1] * line_two[0, 0], line_one[0, 1] * line_two[0, 1]],
            [line_three[0, 0] * line_four[0, 0], line_three[0, 0] * line_four[0, 1] + line_three[0, 1] * line_four[0, 0], line_three[0, 1] * line_four[0, 1]],
            [0, 0, 1]
        ])

        print(A)

        b = np.matrix([[0,0,1]])

        x = np.linalg.solve(A, b.T)

        #c, low = scipy.linalg.cho_factor(A)
        #s = scipy.linalg.cho_solve((c, low), b)
        #print(c, low)
        #print(s)

        
        #u, s, v = np.linalg.svd(A)
        #c = np.dot(u.T,b.T)
        #w = np.linalg.solve(np.diag(s),c)
        #x = np.dot(v.T,w)

        print("x: \n", x)

        KK_T = np.matrix([
            [x[0, 0], x[1, 0]],
            [x[1, 0], 1.0]
        ])

        print("KK.T matrix: \n", KK_T)

        K = scipy.linalg.cholesky(KK_T, lower = False)

        print("K matrix: \n", K)

        affine_homography_matrix = np.matrix([
            [K[0, 0], K[0, 1], 0],
            [K[1, 0], K[1, 1], 0],
            [0, 0, 1]
        ])

        """affine_homography_matrix = np.matrix([
            [x[0, 0], x[1, 0], 0],
            [0, 1.0, 0],
            [0, 0, 1]
        ])"""

        print(affine_homography_matrix)

        output_image_array = self.transform(affine_homography_matrix, np.array(self.image_data))
        self.image_data = output_image_array

        output_image = Image.fromarray(output_image_array.astype('uint8'))

        #output_image.show()
        print("Drawing")
        self.draw_result(output_image, None)





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


    def uptade_interface_to_step_two(self):
        print("Updating Interface")

        self.current_step = 2

        self.transform_button["text"] = "Go To Step 3"
        self.transform_button["command"] = self.transform_to_similarity
        #self.transform_button.pack()

        self.instructions_label["text"] = "Select two 90 degree angles"
        #self.instructions_label.pack()

        # Reset self.clicked_points_list
        self.clicked_points_list = []

        self.run()



    def select_image(self):
        self.image_file_name = filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("image files","*.jpg;*.png"), ("all files","*.*"))) 
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

    def calculate_homography(self):
        pass

    def draw_result(self, output_image, update_interface):       
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        self.image_data = output_image

        self.image_canvas = tkinter.Canvas(self.image_frame, width=self.image_data.size[0], height=self.image_data.size[1])
        self.image_canvas.pack()
        
        image_tk = ImageTk.PhotoImage(self.image_data)
        self.image_canvas.create_image(self.image_data.size[0]//2, self.image_data.size[1]//2, image=image_tk)

        self.image_canvas.bind("<Button-1>", self.get_click_position)

        if update_interface != None:
            print("Updating")
            update_interface()

        self.run()

application = ApplicationUI()
application.run()