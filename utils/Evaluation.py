
from sklearn import metrics
import numpy as np
import cv2


class Evaluation():
    """Compute pixel-level confusion counts and derived change-detection metrics."""
    def __init__(self, label, pred):
        super(Evaluation, self).__init__()
        self.label = label / 255
        self.pred = pred / 255
        raw = self.label.shape[0]
        col = self.label.shape[1]
        size = raw * col
        union = np.clip(((self.label + self.pred)), 0, 1)
        intersection = (self.label * self.pred)
        self.TP = int(intersection.sum())
        self.TN = int(size - union.sum())
        self.FP = int((self.pred - intersection).sum())
        self.FN = int((self.label - intersection).sum())

        # Intersection size for the changed class equals TP.
        self.c_num_or = int(union.sum())
        # Intersection size for the unchanged class equals TN.
        self.uc_num_or = int(size - intersection.sum())

    def matrix(self):
        """Return TP, TN, FP, and FN counts."""
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN
        return TP, TN, FP, FN

    def Classification_indicators(self):
        """Return overall accuracy, Cohen's kappa, and average accuracy."""
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        OA = (TP + TN) / (TP + TN + FP + FN)
        kappa = metrics.cohen_kappa_score((self.label > 0.1).astype(int).flatten(), self.pred.flatten())
        AA = (TP / (TP + FN) + TN / (TN + FP)) / 2

        return OA*100, kappa*100, AA*100

    def CD_indicators(self):
        """Return false-alarm, missed-alarm, and total-error rates."""
        # Binary convention: 0 denotes unchanged and 1 denotes changed.
        #                    0  1(pred)
        # [[a,b]   (label) 0[TP,FP]
        # [c, d]]          1[FN,TN]
        ######################################
        # Report_ConfusionMat = metrics.confusion_matrix(y_true=self.label.flatten(), y_pred=self.pred.flatten())
        # if Report_ConfusionMat.size == 1:
            # return 0.0, 0.0, 0.0
        # else:
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        FA = FP / (FP + TN)
        MA = FN / (FN + TP)
        TE = (FP + FN) / (TP + TN + FP + FN)

        return FA*100, MA*100, TE*100

    # Completeness (recall), correctness (precision), and Jaccard quality.
    def Landsilde_indicators(self):
        """Return completeness, correctness, and quality for changed pixels."""
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        Completeness = TP / (TP + FN)
        Correctness = TP / (TP + FP)
        Quality = TP / (TP + FP + FN)

        return Completeness*100, Correctness*100, Quality*100

    # Return mean IoU and class-wise IoU values.
    def IOU_indicator(self):
        c_num_and = self.c_num_and
        c_num_or = self.c_num_or
        uc_num_and = self.uc_num_and
        uc_num_or = self.uc_num_or

        c_iou = (c_num_and / c_num_or) * 100
        uc_iou = (uc_num_and / uc_num_or) * 100
        mIoU = (c_iou + uc_iou) / 2

        return mIoU, c_iou, uc_iou


    # Return precision, recall, and F1 for the changed class.
    def ObjectExtract_indicators(self):
        """Return changed-class precision, recall, and F1 score."""
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        # OA = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = (2 * Precision * Recall) / (Precision + Recall)

        return Precision*100, Recall*100, F1*100



if __name__ == "__main__":
    pred_path = "D:\\PHD_Research\\DL\\U-Net\\data\\AriealData\\test\\image\\0_res.png"
    label_path = "D:\\PHD_Research\\DL\\U-Net\\data\\AriealData\\test\\label\\0.tif"
    pred = cv2.imread(pred_path)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    Indicators = Evaluation(label, pred)
    OA, kappa, AA = Indicators.Classification_indicators()
    FA, MA, TE = Indicators.CD_indicators()
    CP, CR, AQ = Indicators.Landsilde_indicators()
    IOU = Indicators.IOU_indicator()
    Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
    print("(OA, KC, AA)", OA, kappa, AA)
    print("(FA, MA, TE)", FA, MA, TE)
    print("(CP, CR, AQ)", CP, CR, AQ)
    print("(IoU, Precision, Recall, F1-score)", IOU, Precision, Recall, F1)
