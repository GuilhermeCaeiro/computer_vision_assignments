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
            [6, 35, 1],
            [493, 368, 1],
            [41, 155, 1],
            [291, 24, 1],

            
            
            [6, 35, 1],
            [493, 368, 1],
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

        deformed_points = np.matrix([
            [22, 159, 1],
            [427, 202, 1],
            [153, 168, 1],
            [151, 473, 1],

            
            
            [414, 266, 1],
            [703, 290, 1],
            [536, 223, 1],
            [535, 618, 1],
            

            
            [383, 746, 1],
            [857, 656, 1],
            [798, 411, 1],
            [795, 715, 1],
            


            [765, 456, 1],
            [1007, 447, 1],
            [988, 388, 1],
            [985, 707, 1],
            

            
            #[87, 721, 1],
            #[405, 673, 1],
            #[229, 553, 1],
            #[229, 769, 1],

            [702, 6, 1],
            [1039, 96, 1],
            [800, 10, 1],
            [795, 758, 1],
        ])

        deformed_points = np.matrix([
            [154.37, 451.36, 1.0],
            [152.575, 21.4933, 1.0],
            [64.62, 320.609, 1.0],
            [403.875, 336.729, 1.0],
            
            [256.685, 37.6133, 1.0],
            [755.695, 128.96, 1.0],
            [534.91, 91.3467, 1.0],
            [534.91, 340.311, 1.0],
            
            [281.815, 617.933, 1.0],
            [723.385, 567.782, 1.0],
            [696.46, 347.476, 1.0],
            [696.46, 515.84, 1.0],

            [709.025, 343.893, 1.0],
            [705.435, 120.00, 1.0],
            [538.5, 211.351, 1.0],
            [707.23, 232.844, 1.0],
            
            [337.46, 229.262, 1.0],
            [538.5, 248.964, 1.0],
            [337.46, 327.773, 1.0],
            [335.665, 118.213, 1.0],
        ])


        # TODO Uncommet
        #deformed_points = self.clicked_points_list

        lines = []

        for i in range(0, 20, 2):
            point_1 = deformed_points[i]
            point_2 = deformed_points[i + 1]

            line = np.cross(point_1, point_2)

            line = (line * (1 / line[0, 2]))

            lines.append(line)

        print("Lines: \n", lines)

        system_of_equations = []
        b = []

        for i in range(0, 10, 2):
            l = lines[i]
            m = lines[i + 1]

            system_of_equations.append([
                (l[0, 0] * m[0, 0]), 
                ((l[0, 0] * m[0, 1]) + (l[0, 1] * m[0, 0])) / 2,
                (l[0, 1] * m[0, 1]),
                ((l[0, 0] * m[0, 2]) + (l[0, 2] * m[0, 0])) / 2,
                ((l[0, 1] * m[0, 2]) + (l[0, 2] * m[0, 1])) / 2,
                #(l[0, 2] * m[0, 2]),
                #1.0
            ])

            b.append(  (l[0, 2] * m[0, 2]))

        #system_of_equations.append([0, 0, 0, 0, 0, 0]) # last number is 1 to force f = 1
        #b.append(0)

        print("System of equations: \n", system_of_equations)
        print("b: ", b)
        system_of_equations = np.matrix(system_of_equations)

        #b = np.matrix([[0, 0, 0, 0, 0, 1]]) # last number is 1 to force f = 1
        b = np.matrix([b])

        print(system_of_equations.shape, b.shape)

        #x = np.linalg.inv(system_of_equations) * b.T
        #x = np.linalg.solve(system_of_equations, b.T)
        x = np.linalg.lstsq(system_of_equations, b.T)[0]

        print("x: \n", x)

        conic = np.matrix([
            [x[0, 0], x[1, 0] / 2, x[3, 0] / 2],
            [x[1, 0] / 2, x[2, 0], x[4, 0] / 2],
            [x[3, 0] / 2, x[4, 0] / 2, 1.0]
        ])

        print("Conic matrix: \n", conic)

        w, v = np.linalg.eig(conic)

        print("Eigenvalues: \n", w)
        print("Eigenvectors: \n", v)

        """

        KK_T = np.matrix([
            [x[0, 0], x[1, 0] / 2],
            [x[1, 0] / 2, x[2, 0]],
        ])

        print("KK.T matrix: \n", KK_T)

        KK_T_v = np.matrix([
            [x[3, 0] / 2],
            [x[4, 0] / 2],
        ])

        print("KK.T * v matrix: \n", KK_T_v)

        v = np.linalg.inv(KK_T) * KK_T_v
        #v = np.linalg.lstsq(KK_T, KK_T_v)[0]

        print("v matrix: \n", v)

        K = scipy.linalg.cholesky(KK_T, lower = False)
        #K = np.linalg.qr(KK_T, "r")

        print("K matrix: \n", K)


        projective_homography = np.matrix([
            [1, 0, 0],
            [0, 1, 0],
            [v[0, 0], v[1, 0], 1],
        ])

        print("Projective homography: \n", projective_homography)

        affine_homography = np.matrix([
            [K[0, 0], K[0, 1], 0],
            [K[1, 0], K[1, 1], 0],
            [0, 0, 1],
        ])

        print("Affine homography: \n", affine_homography)


        # matrix to transform from projective space to similarity ()
        projective_to_similarity_homography_matrix = projective_homography * affine_homography
        #"""

        #"""
        # Based on https://github.com/jimmysoda/imgrectifier/blob/master/proj2metric_circle.m

        #w, v = np.linalg.eig(conic)

        #print("Eigenvalues: \n", w)
        #print("Eigenvectors: \n", v)

        v_transp = v.T
        #w[0] = w[0] * -1 if w[0] < 0 else w[0]
        #w[1] = w[1] * -1 if w[1] < 0 else w[1]

        print(w)

        #ss = np.matrix([
        #    [np.sqrt(w[0]), 0, 0],
        #    [0, np.sqrt(w[1]), 0],
        #    [0, 0, 10]
        #])

        ss = np.diag(w)
        print(ss)

        v_transp = ss * v_transp
        v = v_transp

        print(v)
        #v = np.linalg.inv(v)
        projective_to_similarity_homography_matrix = v
        #"""

        """

        KK_T = np.matrix([
            [x[0, 0], x[1, 0] / 2],
            [x[1, 0] / 2, x[2, 0]],
        ])

        print("KK.T matrix: \n", KK_T)

        KK_T_v = np.matrix([
            [x[3, 0] / 2],
            [x[4, 0] / 2],
        ])

        print("KK.T * v matrix: \n", KK_T_v)

        v = np.linalg.inv(KK_T) * KK_T_v
        #v = np.linalg.lstsq(KK_T, KK_T_v)[0]

        print("v matrix: \n", v)

        U, S, V = np.linalg.svd(KK_T)
        
        print("U", U)
        print("S", S)
        #S = np.asmatrix(S).T
        S = np.matrix([[S[0], 0], [0, S[1]]])
        print("S", S)
        print("V", V)

        K = U * np.sqrt(S)
        K = K * V ###


        print(K)

        projective_homography = np.matrix([
            [1, 0, 0],
            [0, 1, 0],
            [v[0, 0], v[1, 0], 1],
        ])

        print("Projective homography: \n", projective_homography)

        affine_homography = np.matrix([
            [K[0, 0], K[0, 1], 0],
            [K[1, 0], K[1, 1], 0],
            [0, 0, 1],
        ])

        print("Affine homography: \n", affine_homography)

        # matrix to transform from projective space to similarity ()
        projective_to_similarity_homography_matrix = projective_homography * affine_homography

        #"""

        # Draw lines. Only for debugging purposes, because the self.run will stop the execution of the remainder of this method.
        #self.image_canvas.create_line(deformed_points[0, 0], deformed_points[0, 1], deformed_points[1, 0], deformed_points[1, 1], width=5, fill='green')
        #self.image_canvas.create_line(deformed_points[2, 0], deformed_points[2, 1], deformed_points[3, 0], deformed_points[3, 1], width=5, fill='green')
        #self.image_canvas.create_line(deformed_points[4, 0], deformed_points[4, 1], deformed_points[5, 0], deformed_points[5, 1], width=5, fill='blue')
        #self.image_canvas.create_line(deformed_points[6, 0], deformed_points[6, 1], deformed_points[7, 0], deformed_points[7, 1], width=5, fill='blue')
        #self.image_canvas.create_line(deformed_points[8, 0], deformed_points[8, 1], deformed_points[9, 0], deformed_points[9, 1], width=5, fill='red')
        #self.image_canvas.create_line(deformed_points[10, 0], deformed_points[10, 1], deformed_points[11, 0], deformed_points[11, 1], width=5, fill='red')
        #self.image_canvas.create_line(deformed_points[12, 0], deformed_points[12, 1], deformed_points[13, 0], deformed_points[13, 1], width=5, fill='orange')
        #self.image_canvas.create_line(deformed_points[14, 0], deformed_points[14, 1], deformed_points[15, 0], deformed_points[15, 1], width=5, fill='orange')
        #self.image_canvas.create_line(deformed_points[16, 0], deformed_points[16, 1], deformed_points[17, 0], deformed_points[17, 1], width=5, fill='yellow')
        #self.image_canvas.create_line(deformed_points[18, 0], deformed_points[18, 1], deformed_points[19, 0], deformed_points[19, 1], width=5, fill='yellow')
        #self.run()
        

        print("Projective to Similarity Homography: \n", projective_to_similarity_homography_matrix)

        output_image_array = self.transform(projective_to_similarity_homography_matrix, np.array(self.image_data))
        self.image_data = output_image_array

        output_image = Image.fromarray(output_image_array.astype('uint8'))
        output_image.save("projective_to_similarity.bmp")
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
        output_width = 768#1024
        output_height = int((total_y/total_x) * output_width)
        #output_height = 768

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
            maximun_points = 20
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