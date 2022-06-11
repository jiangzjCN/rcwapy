import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Model:

    def __init__(self, eps_background, grid, period):
        self.period = period
        self.num_layer = 0
        self.thickness = []
        self.grid = grid
        self.shadow = np.ones((grid[1], grid[0], 0), dtype='float64') * -1
        self.model = np.zeros((grid[1], grid[0], 0), dtype='cdouble')
        self.is_homo = np.zeros(0, dtype='bool')
        self.background = np.zeros(2, dtype='int')
        self.eps_gnd = np.zeros(2, dtype='cdouble')
        self.history = np.empty((0, 7), dtype='object')
        self.step = 0
        if eps_background[0] != eps_background[1]:
            self.num_material = 2
            self.material = np.empty(2, dtype='object')
            self.material[0], self.background[0] = eps_background[0], 0
            self.material[1], self.background[1] = eps_background[1], 1
        else:
            self.material = np.empty(1, dtype='object')
            self.num_material = 1
            self.material[0] = eps_background[0]
        self.template = np.empty(
            (grid[1], grid[0], self.num_layer, self.num_material), dtype='bool')

    def add_layer(self, thickness):
        self.num_layer += 1
        self.thickness.append(thickness)
        self.shadow = np.dstack(
            (self.shadow, -1 * np.ones((self.grid[1], self.grid[0]))))
        self.model = np.dstack(
            (self.model, np.zeros((self.grid[1], self.grid[0]))))
        self.template = np.concatenate((self.template, np.zeros((self.grid[1], self.grid[0], 1, self.num_material),
                                                                dtype='bool')), axis=2)
        self.is_homo = np.concatenate(
            (self.is_homo, np.zeros(1, dtype='bool')))

    def add_circle(self, x0, y0, radius1, radius2, rotation, material, layer_number):
        self.step += 1
        self.history = np.vstack((self.history, np.asarray([x0, y0, radius1, radius2, rotation, material, layer_number],
                                                           dtype='object')))
        if material in self.material:
            material_mark = np.nonzero(material == self.material)[0][0]
        else:
            self.num_material += 1
            material_mark = self.num_material - 1
            self.material = np.concatenate(
                (self.material, np.array(material, ndmin=1)))
            self.template = np.concatenate((self.template, np.zeros((self.grid[1], self.grid[0], self.num_layer, 1),
                                                                    dtype='bool')), axis=3)
        rotation = np.deg2rad(rotation)
        x = np.linspace(0, self.period[0], self.grid[0])
        y = np.linspace(0, self.period[1], self.grid[1])
        x, y = np.meshgrid(x, y)
        x = x - x0
        y = y - y0
        is_inside = (x * np.cos(rotation) + y * np.sin(rotation)) ** 2 / radius1 ** 2 + \
                    (-x * np.sin(rotation) + y *
                     np.cos(rotation)) ** 2 / radius2 ** 2
        temp = self.shadow[:, :, layer_number - 1]
        temp[is_inside <= 1] = material_mark
        self.shadow[:, :, layer_number - 1] = temp

    def add_rectangle(self, x0, y0, length, width, rotation, material, layer_number):
        self.step += 1
        self.history = np.vstack((self.history, np.asarray([x0, y0, length, width, rotation, material, layer_number],
                                                           dtype='object')))
        if material in self.material:
            material_mark = np.nonzero(material == self.material)[0][0]
        else:
            self.num_material += 1
            material_mark = self.num_material - 1
            self.material = np.concatenate(
                (self.material, np.array(material, ndmin=1)))
            self.template = np.concatenate((self.template, np.zeros(
                (self.grid[1], self.grid[0], self.num_layer, 1), dtype='bool')), axis=3)
        rotation = np.deg2rad(rotation)
        x = np.linspace(0, self.period[0], self.grid[0])
        y = np.linspace(0, self.period[1], self.grid[1])
        x, y = np.meshgrid(x, y)
        x = x - x0
        y = y - y0
        temp_x = x
        x = x * np.cos(rotation) - y * np.sin(rotation)
        y = temp_x * np.sin(rotation) + y * np.cos(rotation)
        temp = self.shadow[:, :, layer_number - 1]
        is_inside = np.logical_and(
            x <= length / 2, x >= -length / 2) & np.logical_and(y <= width / 2, y >= -width / 2)
        temp[is_inside] = material_mark
        self.shadow[:, :, layer_number - 1] = temp

    def check_model(self):
        for i in range(self.num_layer):
            assert - \
                1 not in self.shadow[:, :, i], 'WARNING: Layer' + \
                str(i + 1) + ' is not completed'

    def gen_template(self):
        self.check_model()
        for i in range(self.num_material):
            self.template[:, :, :, i] = (self.shadow == i)
        for i in range(self.num_layer):
            if np.amax(self.shadow[:, :, i]) == np.amin(self.shadow[:, :, i]):
                self.is_homo[i] = True

    def gen_model(self, wavelength):
        for i in range(self.num_material):
            if isinstance(self.material[i], str):
                eps = self.get_eps(self.material[i], wavelength)
                self.model[self.template[:, :, :, i]] = eps
                self.eps_gnd[self.background == i] = eps
            else:
                self.model[self.template[:, :, :, i]] = self.material[i]
                self.eps_gnd[self.background == i] = self.material[i]

    def show_model(self, layer_number=None):
        gridx = np.linspace(0, self.period[0], self.grid[0])
        gridy = np.linspace(0, self.period[1], self.grid[1])
        if not layer_number:
            for i in range(self.num_layer):
                ax = plt.subplot(int(self.num_layer / 3) + 1, 3, i + 1)
                ax.contourf(gridx, gridy, self.shadow[:, :, i], 100)
                ax.axis('equal')
                ax.set_title('Layer %d \n Thickness: %d μm' %
                             (i + 1, self.thickness[i]))
                ax.set_xlabel('x (μm)')
                ax.set_ylabel('y (μm)')
            plt.show()
        else:
            ax = plt.subplot(1, 1, 1)
            ax.contourf(gridx, gridy, self.shadow[:, :, layer_number - 1], 100)
            ax.axis('equal')
            ax.set_title('Layer %d \n Thickness: %d μm' %
                         (layer_number, self.thickness[layer_number - 1]))
            ax.set_xlabel('x (μm)')
            ax.set_ylabel('y (μm)')
            plt.show()

    @staticmethod
    def get_eps(material, wavelength):
        fn = "D:\\Research\\MaterialsData\\"
        data = pd.read_table(fn + material + '.txt', sep='\t', header=None)
        data = data.values
        assert np.amax(data[:, 0]) > wavelength > np.amin(
            data[:, 0]), 'Wavelength is out of range'
        n = np.interp(wavelength, data[:, 0], data[:, 1])
        k = np.interp(wavelength, data[:, 0], data[:, 2])
        return (n - 1j * k) ** 2
