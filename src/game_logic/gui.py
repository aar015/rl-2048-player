import tkinter
import csv
import numpy

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
                            32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
                            512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"}
CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
                   32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
                   512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2"}
FONT = ("Verdana", 40, "bold")

KEY_NEXT = "'d'"


class GameGrid(tkinter.Frame):
    def __init__(self, file):
        tkinter.Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.step = 0
        self.goCounter = 0
        self.game = self.read_file(file)

        self.grid_cells = []
        self.init_grid()
        self.update_grid_cells()

        self.mainloop()

    def init_grid(self):
        background = tkinter.Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = tkinter.Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY,
                                     width=SIZE / GRID_LEN, height=SIZE / GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = tkinter.Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY,
                                  justify=tkinter.CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        if(self.step == numpy.shape(self.game)[0]):
            return
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = 2**self.game[self.step][4 * i + j]
                if new_number == 1:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        self.step += 1
        self.update_idletasks()

    def key_down(self, event):
        if(self.goCounter <= 5):
            self.goCounter += 1
        else:
            self.update_grid_cells()

    def read_file(self, file):
        with open(file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            row_count = sum(1 for row in csv_reader)
            game = numpy.zeros((row_count, 16), dtype=numpy.int)
            csv_file.seek(0)
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                for col in range(len(row)):
                    game[line_count][col] = row[col]
                line_count += 1
        return game
