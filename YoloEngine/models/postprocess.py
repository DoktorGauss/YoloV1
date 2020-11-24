predict_class = y_pred[..., :C]  # ? * 7 * 7 * 20
        predict_trust = y_pred[..., C+B*4:]  # ? * 7 * 7 * 2
        predict_box = y_pred[..., C:C+B*4]  # ? * 7 * 7 * 8

        _label_box = kb.reshape(label_box, [-1, S[0], S[1], 1, 4])
        _predict_box = kb.reshape(predict_box, [-1, S[0], S[1], B, 4])

        label_xy, label_wh = yolo_head(_label_box,inputShape)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        label_xy = kb.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_wh = kb.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2