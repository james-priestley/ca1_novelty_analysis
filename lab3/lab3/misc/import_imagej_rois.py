import sima
import read_roi
import numpy as np

def imagej2sima(roi_zip_path, im_shape=None):
    rois = []
    rlist = read_roi.read_roi_zip(roi_zip_path)
    for label, imagej_roi in rlist.items():
        x_coords = imagej_roi['x']
        y_coords = imagej_roi['y']
        z = imagej_roi['position']

        n_vertices = len(x_coords)

        polygon = np.zeros((n_vertices, 3))
        polygon[:, 0] = x_coords
        polygon[:, 1] = y_coords
        polygon[:, 2] = z - 1

        roi = sima.ROI.ROI(polygons=polygon, label=label, id=label, im_shape=im_shape)
        rois.append(roi)

    print("{} ROIs loaded from ImageJ.".format(len(rlist)))

    return sima.ROI.ROIList(rois)
