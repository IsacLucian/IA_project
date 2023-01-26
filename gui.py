import customtkinter
import os
from PIL import Image
from tkinter import filedialog as fd
from backend import svm, decision_tree, random_forest, logistic_regression, naive_bayes, knn, ensemble


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.upload_frame_filepath = None
        self.title("Prediction of personality traits through language.py")
        self.geometry("700x450")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        image_path = os.path.join(os.curdir, "images")
        self.logo_image = customtkinter.CTkImage(dark_image=Image.open(os.path.join(image_path, "logo_dark.png")),
                                                 light_image=Image.open(os.path.join(image_path, "logo_light.png")),
                                                 size=(200, 80))
        self.home_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "home_dark.png")),
                                                 dark_image=Image.open(os.path.join(image_path, "home_light.png")),
                                                 size=(20, 20))
        self.text_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "T_dark.png")),
                                                 dark_image=Image.open(os.path.join(image_path, "T_light.png")),
                                                 size=(20, 20))
        self.upload_image = customtkinter.CTkImage(
            light_image=Image.open(os.path.join(image_path, "upload_dark.png")),
            dark_image=Image.open(os.path.join(image_path, "upload_light.png")),
            size=(20, 20))

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame,
                                                             image=self.logo_image,
                                                             text="")
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame,
                                                   corner_radius=0,
                                                   height=40,
                                                   border_spacing=10,
                                                   text="Home",
                                                   fg_color="transparent",
                                                   text_color=("gray10", "gray90"),
                                                   hover_color=("gray70", "gray30"),
                                                   image=self.home_image,
                                                   anchor="w",
                                                   command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.text_button = customtkinter.CTkButton(self.navigation_frame,
                                                   corner_radius=0,
                                                   height=40,
                                                   border_spacing=10,
                                                   text="Text",
                                                   fg_color="transparent",
                                                   text_color=("gray10", "gray90"),
                                                   hover_color=("gray70", "gray30"),
                                                   image=self.text_image,
                                                   anchor="w",
                                                   command=self.text_button_event)
        self.text_button.grid(row=2, column=0, sticky="ew")

        self.upload_button = customtkinter.CTkButton(self.navigation_frame,
                                                     corner_radius=0,
                                                     height=40,
                                                     border_spacing=10,
                                                     text="File",
                                                     fg_color="transparent",
                                                     text_color=("gray10", "gray90"),
                                                     hover_color=("gray70", "gray30"),
                                                     image=self.upload_image,
                                                     anchor="w",
                                                     command=self.upload_button_event)
        self.upload_button.grid(row=3, column=0, sticky="ew")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame,
                                                                values=["Light", "Dark"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        self.home_frame_title_label = customtkinter.CTkLabel(self.home_frame,
                                                             text="\nPrediction of Personality Traits Through\n"
                                                                  "Language\n",
                                                             font=("Calibri", 20))
        self.home_frame_title_label.grid(row=0, column=0, padx=20, pady=10)

        self.home_frame_description = customtkinter.CTkTextbox(self.home_frame,
                                                               width=350,
                                                               height=198,
                                                               border_width=2,
                                                               wrap=customtkinter.WORD)
        self.home_frame_description.grid(row=1, column=0, padx=20, pady=10)
        self.home_frame_description.insert("insert",
                                           text="Welcome!\nThis is the \"Prediction of Personality Traits "
                                                "Through Language\" project, developed by Carausu Madalina, Haiura "
                                                "Isabela, Isac Lucian, Stoica Dragos and Petrea Daniela at "
                                                "the Faculty of Computer Science at UAIC. Our artificial "
                                                "intelligence project aims to provide users with the ability"
                                                " to predict the personality traits of individuals based on "
                                                "their language use. Simply type in a text or upload a text "
                                                "file, select the model you want to use, and our application "
                                                "will predict the personality traits of the individual. "
                                                "\nThank you for using our app!")
        self.home_frame_description.configure(state="disabled")

        # create text frame
        self.text_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.text_frame.grid_rowconfigure(6, weight=1)
        self.text_frame.grid_columnconfigure(2, weight=1)
        self.text_frame_title_label = customtkinter.CTkLabel(self.text_frame,
                                                             text="\nPrediction of Personality Traits Through\n"
                                                                  "Language\n",
                                                             font=("Calibri", 20))
        self.text_frame_title_label.grid(row=0, column=0, columnspan=2, padx=0, pady=0)

        self.text_frame_input_label = customtkinter.CTkLabel(self.text_frame,
                                                             text="Enter your text here: ")
        self.text_frame_input_label.grid(row=1, column=0, padx=20, pady=0, sticky="nw")

        self.text_frame_input_text_box = customtkinter.CTkTextbox(self.text_frame,
                                                                  width=200,
                                                                  height=100,
                                                                  border_width=2,
                                                                  corner_radius=10)
        self.text_frame_input_text_box.grid(row=2, column=0, padx=20, pady=0, sticky="w")

        self.text_frame_model_label = customtkinter.CTkLabel(self.text_frame,
                                                             text="Choose model: ")
        self.text_frame_model_label.grid(row=3, column=0, padx=20, pady=0, sticky="w")
        self.text_frame_model_menu = customtkinter.CTkOptionMenu(self.text_frame,
                                                                 values=["SVM", "Decision Tree", "Random Forest", "Logistic "
                                                                                                         "Regression",
                                                                         "Naive Bayes", "KNN", "Ensemble"])
        self.text_frame_model_menu.grid(row=4, column=0, padx=20, pady=0, sticky="w")

        self.text_frame_submit_button = customtkinter.CTkButton(self.text_frame,
                                                                text="Submit",
                                                                border_spacing=10,
                                                                command=self.submit_text_event)
        self.text_frame_submit_button.grid(row=6, column=0, padx=20, pady=0, sticky="w")

        self.text_frame_response_label = customtkinter.CTkLabel(self.text_frame,
                                                                text="Personality prediction: ")
        self.text_frame_response_label.grid(row=1, column=1, padx=0, pady=0, sticky="n")

        self.text_frame_text_box = customtkinter.CTkTextbox(self.text_frame,
                                                            width=180,
                                                            height=260,
                                                            border_width=2,
                                                            corner_radius=10,
                                                            fg_color=("#bfbfbf", "#343638"))
        self.text_frame_text_box.configure(state="disabled")
        self.text_frame_text_box.grid(row=2, column=1, rowspan=5, padx=10, pady=0, sticky="n")

        # create upload file frame
        self.upload_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.upload_frame.grid_rowconfigure(6, weight=1)
        self.upload_frame.grid_columnconfigure(2, weight=1)
        self.upload_frame_title_label = customtkinter.CTkLabel(self.upload_frame,
                                                               text="\nPrediction of Personality Traits Through\n"
                                                                    "Language\n",
                                                               font=("Calibri", 20))
        self.upload_frame_title_label.grid(row=0, column=0, columnspan=2, padx=0, pady=0)

        self.upload_frame_file_button = customtkinter.CTkButton(self.upload_frame, text="Choose file",
                                                                command=self.upload_file_event)
        self.upload_frame_file_button.grid(row=2, column=0, padx=20, pady=0, sticky="w")

        self.upload_frame_filename_label = customtkinter.CTkLabel(self.upload_frame, text="")

        self.upload_frame_model_label = customtkinter.CTkLabel(self.upload_frame,
                                                               text="Choose model: ")
        self.upload_frame_model_label.grid(row=4, column=0, padx=20, pady=0, sticky="w")

        self.upload_frame_model_menu = customtkinter.CTkOptionMenu(self.upload_frame,
                                                                   values=["SVM", "Decision Tree", "Random Forest", "Logistic "
                                                                                                           "Regression",
                                                                           "Naive Bayes", "KNN", "Ensemble"])
        self.upload_frame_model_menu.grid(row=5, column=0, padx=20, pady=0, sticky="w")

        self.upload_frame_submit_button = customtkinter.CTkButton(self.upload_frame,
                                                                  text="Submit",
                                                                  border_spacing=10,
                                                                  command=self.submit_upload_event)
        self.upload_frame_submit_button.grid(row=6, column=0, padx=20, pady=(70, 0), sticky="w")

        self.upload_frame_response_label = customtkinter.CTkLabel(self.upload_frame,
                                                                  text="Personality prediction: ")
        self.upload_frame_response_label.grid(row=1, column=1, padx=(70, 0), pady=0, sticky="n")

        self.upload_frame_text_box = customtkinter.CTkTextbox(self.upload_frame,
                                                              width=180,
                                                              height=260,
                                                              border_width=2,
                                                              corner_radius=10,
                                                              fg_color=("#bfbfbf", "#343638"))
        self.upload_frame_text_box.configure(state="disabled")
        self.upload_frame_text_box.grid(row=2, column=1, rowspan=5, padx=(70, 0), pady=0, sticky="n")

        # select default frame
        self.select_frame_by_name("home")

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.text_button.configure(fg_color=("gray75", "gray25") if name == "text" else "transparent")
        self.upload_button.configure(fg_color=("gray75", "gray25") if name == "upload" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "text":
            self.text_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.text_frame.grid_forget()
        if name == "upload":
            self.upload_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.upload_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def text_button_event(self):
        self.select_frame_by_name("text")

    def upload_button_event(self):
        self.select_frame_by_name("upload")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def upload_file_event(self):
        filepath = fd.askopenfilename(filetypes=[("Text files", ".txt")])
        self.upload_frame_filepath = filepath
        self.upload_frame_filename_label.configure(text=str(filepath).split('/')[-1])
        self.upload_frame_filename_label.grid(row=3, column=0, padx=20, pady=0, sticky="w")

    def submit_text_event(self):
        text = self.text_frame_input_text_box.get('1.0', 'end')
        model = self.text_frame_model_menu.get()

        self.text_frame_input_text_box.delete('1.0', 'end')
        self.text_frame_text_box.configure(state="normal")
        self.text_frame_text_box.delete('1.0', 'end')

        response = self.call_corresponding_be_function(model, text)

        self.text_frame_text_box.insert("insert", response)
        self.text_frame_text_box.configure(state="disabled")

    def submit_upload_event(self):
        model = self.upload_frame_model_menu.get()
        text = ""
        if self.upload_frame_filepath:
            with open(self.upload_frame_filepath, "r") as f:
                for line in f:
                    text += line
        self.upload_frame_text_box.configure(state="normal")
        self.upload_frame_text_box.delete('1.0', 'end')

        response = self.call_corresponding_be_function(model, text)

        self.upload_frame_text_box.insert("insert", response)
        self.upload_frame_text_box.configure(state="disabled")

    def call_corresponding_be_function(self, model, text):
        if model == "SVM":
            return svm(text)
        elif model == "Decision Tree":
            return decision_tree(text)
        elif model == "Random Forest":
            return random_forest(text)
        elif model == "Logistic Regression":
            return logistic_regression(text)
        elif model == "Naive Bayes":
            return naive_bayes(text)
        elif model == "KNN":
            return knn(text)
        elif model == "Ensemble":
            return ensemble(text)


if __name__ == "__main__":
    app = App()
    app.mainloop()
