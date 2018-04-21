"""Retrieve data from NSCLC-Radiomics"""

import os

import pandas as pd
import pydicom as pc

import config

class DataRetriver(object):
    """
    Retrieve data from NSCLC-Radiomics
    """
    def __init__(self, config: config.Config):
        self.config = config
        self.data_path = config.NSCLC_data_path
        self.roi_id = []
        self.roi_contour = []
        self.dcm_id = []
        self.dcm_path = []
        self.dcm_pos = []
        self.data = None

    def start(self):
        self.traverse(self.data_path)
        roi_data = {"id":self.roi_id, "contour":self.roi_contour}
        try:
            roi_data = pd.DataFrame(roi_data)
        except RuntimeError:
            return
        
        dcm_data = {"id":self.dcm_id, "path":self.dcm_path, "pos":self.dcm_pos}
        try:
            dcm_data = pd.DataFrame(dcm_data)
        except RuntimeError:
            return

        self.merge(roi_data, dcm_data)
        self.save()

    def traverse(self, parent):
        """travere dataset"""
        children = os.listdir(parent)
        if  len(children) == 1 and os.path.isfile(os.path.join(parent, children[0])):
            self.roi_retrieve(os.path.join(parent, children[0]))
            return
        
        for child in children:
            child = os.path.join(parent, child)
            if os.path.isdir(child):
                self.traverse(child)
            else:
                self.dcm_retrieve(child)

    def roi_retrieve(self, path):
        """retrieve roi info"""
        dcm = pc.read_file(path)
        try:
            dcm = dcm.ROIContourSequence[0].ContourSequence
        except IndexError:
            return

        for i in range(len(dcm)):
            id = dcm[i].ContourImageSequence[0].ReferencedSOPInstanceUID
            contour = dcm[i].ContourData
            self.roi_id.append(id)
            self.roi_contour.append(contour)
    
    def dcm_retrieve(self, path):
        """retrieve dcm id """
        dcm = pc.read_file(path)
        id = dcm.SOPInstanceUID
        pos = dcm.ImagePositionPatient
        self.dcm_id.append(id)
        self.dcm_path.append(path)
        self.dcm_pos.append(pos)

    
    def merge(self, roi_data, dcm_data):
        """merge two tables(roi_data, dcm_data)"""
        self.data = pd.merge(roi_data, dcm_data, left_on="id", right_on="id", how="left")

    def save(self):
        self.data = self.data.dropna()
        self.data.to_excel(self.config.base_data_path + "data.xlsx")