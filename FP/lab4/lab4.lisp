(defun check-presence-plus (text pos-first-plus)
    (cond ((null text) -1)
          ((position #\+ (first text)) pos-first-plus)
          (t (check-presence-plus (rest text) (1+ pos-first-plus))))
)

(defun replace-digits-to-minus (sentence pos)
	(if (digit-char-p (char sentence pos)) (setf (aref sentence pos) #\-))
)

(defun replace-digits-before-plus (text res-text pos-first-plus pos-sentence)
    (if (null text) (reverse res-text)
        (let ((sentence (copy-seq (first text))))
            (cond ((> pos-first-plus pos-sentence) (loop for i from 0 to (1- (length sentence))
                    do (replace-digits-to-minus sentence i)))
				  ((= pos-first-plus pos-sentence) (loop for i from 0 to (position #\+ sentence)
                    do (replace-digits-to-minus sentence i))))
            (replace-digits-before-plus (rest text) (cons sentence res-text) pos-first-plus (1+ pos-sentence))))
)

(defun get-res-text (text)
    (if (= (check-presence-plus text 0) -1) text
        (replace-digits-before-plus text nil (check-presence-plus text 0) 0))
)