(defun matrix-tr-tl (n)
  (let  ((matrix (make-array (list n n)))
        (num 1))

  (dotimes (i n)
	(if (evenp i)
      (loop for j from (1- n) downto 0
        do (setf (aref matrix i j) num)
           (setf num (1+ num)))
    
	  (dotimes (j n)
	      (setf (aref matrix i j) num)
	      (setf num (1+ num)))))
    matrix)
)

(defun print-matrix (matrix &optional (chars 3) stream)
	;; Предполагаем, что требуется
	;; 3 знака по умолчанию на каждый элемент,
	;; 6 знаков на #2А и скобки.
  (let ((*print-right-margin* (+ 6 (* (1+ chars)   ;плюс пробел
                                      (array-dimension matrix 1)))))
    (pprint matrix stream)
    (values)))

(defun matrix-tl-bl (n)
  (matrix-tr-tl n)
)