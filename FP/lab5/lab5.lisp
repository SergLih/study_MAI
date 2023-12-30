(defclass cart ()                     ; имя класса и надклассы
	((x :initarg :x :reader cart-x)   ; дескриптор слота x
	 (y :initarg :y :reader cart-y)))  ; дескриптор слота y
	
(defclass polar ()
	((radius :initarg :radius :accessor radius) 	; длина >=0
	 (angle  :initarg :angle  :accessor angle)))	; угол (-?;?]

(defmethod cart-x ((p polar))
    (* (radius p) (cos (angle p))))

(defmethod cart-y ((p polar))
    (* (radius p) (sin (angle p))))
	 
(defgeneric to-cart (arg)
  (:documentation "Преобразование аргумента в декартову систему.")
  (:method ((c cart))
    c)
  (:method ((p polar))
    (make-instance 'cart
                   :x (cart-x p)
                   :y (cart-y p))))			   

(defvar *tolerance* 0.0001)

(defun on-line-check (vertice1 vertice2 vertice3)
	(let ((x1 (cart-x vertice1))
          (x2 (cart-x vertice2))
          (y1 (cart-y vertice1))
		  (y2 (cart-y vertice2))
		  (x  (cart-x vertice3))
		  (y  (cart-y vertice3)))
	(<= (/ (abs (+ (- (* (- y2 y1) x) (* (- x2 x1) y)) (- (* x2 y1) (* y2 x1))))
			(sqrt (+ (* (- y2 y1) (- y2 y1)) (* (- x2 x1) (- x2 x1))))) *tolerance*))) 


(defun on-single-line-p (vertices)
	(sort vertices (lambda (p1 p2) (or (< (cart-x p1) (cart-x p2)) (and (= (cart-x p1) (cart-x p2)) (< (cart-y p1) (cart-y p2))))))
    (if (<= (length vertices) 2) T
		(loop for i from 1 to (- (length vertices) 2)
					always (on-line-check (nth 0 vertices) (nth (- (length vertices) 1) vertices) (nth i vertices)))))
						
(defun test-1 ()
    (let (ver1 ver2 ver3)
        (setq ver1 (make-instance 'cart :x 0 :y 0))
        (setq ver2 (make-instance 'cart :x 5 :y 1))
		(setq ver3 (make-instance 'cart :x 2 :y 2))
		(on-single-line-p (list ver1 ver2 ver3))))

(defun test-2 ()
    (let (ver1 ver2)
        (setq ver1 (make-instance 'cart :x 0 :y 0))
		(setq ver2 (make-instance 'cart :x -8 :y -7))
		(on-single-line-p (list ver1 ver2))))

(defun test-3 ()
    (let (ver1)
        (setq ver1 (make-instance 'cart :x -1 :y 1))
		(on-single-line-p (list ver1))))

(defun test-4 ()
    (let (ver1 ver2 ver3 ver4)
		(setq ver1 (make-instance 'cart :x 1 :y 1))
		(setq ver2 (make-instance 'cart :x 5 :y 5))
		(setq ver3 (make-instance 'cart :x -2 :y -2))
		(setq ver4 (make-instance 'cart :x 6 :y 6))
		(on-single-line-p (list ver1 ver2 ver3 ver4))))
	
(defun test-5 ()
    (let (ver1 ver2 ver3 ver4 ver5)
		(setq ver1 (make-instance 'cart :x 1 :y 1))
		(setq ver2 (make-instance 'cart :x 5 :y 5))
		(setq ver3 (make-instance 'cart :x 9 :y -5))
		(setq ver4 (make-instance 'cart :x -2 :y -2))
		(setq ver5 (make-instance 'cart :x 6 :y 6))
		(on-single-line-p (list ver1 ver2 ver3 ver4 ver5))))

(defun test-6 ()
	(let (ver1)		
		(setq ver1 (make-instance 'polar :radius 5 :angle (/ pi 6)))
		(on-single-line-p (list ver1))))
				
(defun test-7 ()
	(let (ver1 ver2 ver3 ver4)		
		(setq ver1 (make-instance 'polar :radius 5 :angle (/ pi 6)))
		(setq ver2 (make-instance 'polar :radius 2 :angle (/ pi 2)))
		(setq ver3 (make-instance 'polar :radius 7 :angle (/ pi 4)))
		(setq ver4 (make-instance 'polar :radius 1 :angle (/ pi 6)))
		(on-single-line-p (list ver1 ver2 ver3 ver4))))
		
(defun test-8 ()
    (let (ver1 ver2 ver3 ver4)
		(setq ver1 (make-instance 'cart :x 555 :y 555))
		(setq ver2 (make-instance 'cart :x 2 :y 2.001))
		(setq ver3 (make-instance 'cart :x 1.0001 :y 1))
		(setq ver4 (make-instance 'cart :x 0 :y 0))
		(on-single-line-p (list ver1 ver2 ver3 ver4))))
		
(defun test-9 ()
	(let (ver1 ver2 ver3)		
		(setq ver1 (make-instance 'polar :radius 1 :angle (/ pi 4)))
		(setq ver2 (make-instance 'polar :radius 1.0001 :angle (+ (/ pi 4) 0.0001)))
		(setq ver3 (make-instance 'polar :radius 5 :angle (/ pi 4)))
		(on-single-line-p (list ver1 ver2 ver3))))