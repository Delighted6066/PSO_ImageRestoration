import numpy as np
from PIL import Image, ImageTk
from tkinter import LabelFrame, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import convolve2d
from tkinter import Tk, Button, Label, filedialog, Entry
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class ImageRestorationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("基于粒子群算法的模糊图像复原系统")

        # 设置窗口大小 宽为屏幕的1/2，高为屏幕的2/3
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 1 / 2)
        window_height = int(screen_height * 2 / 3)
        root.geometry(f"{window_width}x{window_height}")

        # 设置背景颜色
        root.configure(bg="lightblue")

        # 按钮布局
        button_frame = LabelFrame(root, bg="lightblue")
        button_frame.pack(pady=10)

        self.load_button = Button(button_frame, text="加载图像", command=self.load_image, bg="blue", fg="white")
        self.load_button.pack(side="left", padx=10)

        self.restore_button = Button(button_frame, text="复原图像", command=self.restore_image, bg="green", fg="white")
        self.restore_button.pack(side="left", padx=10)

        self.quit_button = Button(button_frame, text="退出系统", command=root.quit, bg="red", fg="white")
        self.quit_button.pack(side="left", padx=10)

        # 粒子群算法参数布局
        parameter_frame = LabelFrame(root, text="粒子群算法参数设置", bg="lightblue")
        parameter_frame.pack(pady=10)

        num_particles_label = Label(parameter_frame, text="粒子数：")
        num_particles_label.pack(side="left")
        self.num_particles_entry = Entry(parameter_frame)
        self.num_particles_entry.pack(side="left", padx=5)

        max_iterations_label = Label(parameter_frame, text="迭代次数：")
        max_iterations_label.pack(side="left")
        self.max_iterations_entry = Entry(parameter_frame)
        self.max_iterations_entry.pack(side="left", padx=5)

        # 图像布局
        image_frame = LabelFrame(root, bg="lightblue")
        image_frame.pack(pady=10)

        self.original_label = Label(image_frame)
        self.original_label.pack(side="left", padx=10)

        self.restored_label = Label(image_frame)
        self.restored_label.pack(side="left", padx=10)

        # 迭代收敛图布局
        convergence_frame = LabelFrame(root, text="迭代收敛图", bg="lightblue")
        convergence_frame.pack(pady=10)

        self.convergence_canvas = None

        self.filename = None
        self.original_image = None
        self.blurred_image = None
        self.restored_image = None

    def load_image(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="选择图像文件",
                                                   filetypes=(("图像文件", ("*.png", "*.jpg", "*.jpeg", "*.bmp")),
                                                              ("所有文件", "*.*")))
        if self.filename:
            self.original_image = load_image(self.filename)
            self.blurred_image = convolve2d(self.original_image, np.ones((5, 5)) / 25, mode='same')
            self.original_label.config(text="原始图像", compound="top")
            self.original_image_label = self.get_image_label(self.original_image)  # 存储原始图像的 Label 对象
            self.original_label.config(image=self.original_image_label)

    def restore_image(self):
        if self.blurred_image is not None:
            mask = np.abs(self.original_image - self.blurred_image)
            num_particles, max_iterations = self.set_particle_parameters()

            restored_image, convergence = restore_image(self.blurred_image, mask, num_particles, max_iterations)

            messagebox.showinfo("复原图像", "复原图像已生成，点击确定查看！")
            self.show_restored_image(restored_image)
            self.show_convergence_plot(convergence)

    def set_particle_parameters(self):
        num_particles = int(self.num_particles_entry.get())
        max_iterations = int(self.max_iterations_entry.get())
        return num_particles, max_iterations

    def show_restored_image(self, restored_image):
        self.restored_label.config(text="复原图像", compound="top")
        self.restored_image_label = self.get_image_label(restored_image)  # 存储复原图像的 Label 对象
        self.restored_label.config(image=self.restored_image_label)

    def show_convergence_plot(self, convergence):
        if self.convergence_canvas is None:
            convergence_frame = LabelFrame(self.root, text="迭代收敛图")
            convergence_frame.pack(pady=10)

            self.convergence_canvas = plt.figure(figsize=(6, 4), dpi=100)
            convergence_canvas = FigureCanvasTkAgg(self.convergence_canvas, master=convergence_frame)
            convergence_canvas.get_tk_widget().pack()

        plt.clf()
        plt.plot(convergence)
        plt.xlabel('迭代次数', fontsize=10)
        plt.ylabel('适应度值', fontsize=10)
        plt.tight_layout()  # 调整子图布局
        self.convergence_canvas.canvas.draw()

    @staticmethod
    def get_image_label(image_array):
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        image_width, image_height = image.size
        if image_width > image_height:
            new_width = 256
            new_height = int((image_height / image_width) * 256)
        else:
            new_width = int((image_width / image_height) * 256)
            new_height = 256
        image = image.resize((new_width, new_height))
        return ImageTk.PhotoImage(image)


def load_image(filename):
    img = Image.open(filename).convert('L')
    img_array = np.array(img) / 255.0  # 转换为灰度图像，并将像素值归一化到[0, 1]
    return img_array


def save_image(filename, img_array):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img.save(filename)


def evaluate_fitness(original, restored):
    # 计算复原图像与原始图像之间的差异
    return np.sum((original - restored) ** 2)


def restore_image(blurred, mask, num_particles, max_iterations):
    image_shape = blurred.shape
    particle_shape = (num_particles, *image_shape)

    # 初始化粒子位置和速度
    particles = np.random.rand(*particle_shape)
    velocities = np.random.rand(*particle_shape)

    global_best_fitness = np.inf
    global_best_particle = None

    convergence = []  # 用于存储每次迭代的适应度值

    # 执行粒子群优化
    for _ in range(max_iterations):
        # 计算每个粒子的当前复原图像
        restored_images = blurred + particles * mask

        # 计算每个粒子的适应度值
        fitness = np.zeros(num_particles)
        for i in range(num_particles):
            fitness[i] = evaluate_fitness(blurred, restored_images[i])

            # 更新全局最佳粒子
            if fitness[i] < global_best_fitness:
                global_best_fitness = fitness[i]
                global_best_particle = restored_images[i]

        # 更新粒子位置和速度
        phi1 = 2.0
        phi2 = 2.0
        inertia = 0.7
        c1 = phi1 * np.random.rand(*particle_shape)
        c2 = phi2 * np.random.rand(*particle_shape)
        velocities = (inertia * velocities +
                      c1 * (global_best_particle - particles) +
                      c2 * (restored_images - particles))
        particles = particles + velocities

        convergence.append(global_best_fitness)  # 将当前适应度值添加到收敛列表中

    return global_best_particle, convergence


if __name__ == "__main__":
    root = Tk()
    gui = ImageRestorationGUI(root)
    root.mainloop()
