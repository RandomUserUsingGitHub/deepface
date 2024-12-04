import numpy as np
import os
import gdown
from typing import List, Tuple, Dict, Any, Union

from deepface.commons import package_utils, folder_utils
from deepface.models.FacialRecognition import FacialRecognition


class ArcFaceClient(FacialRecognition):
    """
    ArcFace model class
    """

    def __init__(self):

        """
        MODELS:
        - glint360k_r100.onnx
        - glint360k_r50_pfc.onnx
        - webface_r50.onnx
        - arcface_ms1mv2_r50.onnx (DEFAULT)
        
        """
        
        home = folder_utils.get_deepface_home()
        file_name = "glint360k_r100.onnx"
        model_file = home + "/.deepface/weights/" + file_name

        self.model = ONNXModelInterface(model_file)
        self.model_name = "ArcFace"
        self.input_shape = (112, 112)
        self.output_shape = 512


class ONNXModelInterface:
    def __init__(
        self,
        model_file: str,  # = "/home/eyz/Documents/SRC/SRC_Pollet/FaRe/models/arcface_ms1mv2_r50.onnx",
        image_size: Tuple[int, int] = (112, 112),
    ):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        self.ort_session = ort.InferenceSession(model_file)
        self.output_names = [self.ort_session.get_outputs()[0].name]
        self.input_name = self.ort_session.get_inputs()[0].name

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        outputs = self.ort_session.run(self.output_names, {self.input_name: imgs})
        return outputs[0]
