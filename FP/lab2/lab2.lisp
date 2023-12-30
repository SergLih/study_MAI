(defun list-traversal(my-list even odd)
  (if (oddp(car my-list))
      (if (> (length(cdr my-list)) 0)
      	(list-traversal (cdr my-list) even (append odd (list (car my-list))))
      	(list even (append odd (list (car my-list)))))
	  (if (> (length(cdr my-list)) 0)
      	(list-traversal (cdr my-list) (append even (list (car my-list))) odd)
      	(list (append even (list (car my-list))) odd)))
  )

(defun my-merge(even odd res)
  (if (and (> (length even) 0) (> (length odd) 0))
      (my-merge (cdr even) (cdr odd) (append res (list (car even)) (list (car odd))))
      res)
  )

(defun even-odd(my-list)
  (if (> (length my-list) 0)
	(let ((res (list-traversal my-list () ())))
		(my-merge (car res) (car (cdr res)) () )))
  )

  
#|
(defun my-merge(even odd res)
  (if (and (> (length even) 0) (> (length odd) 0))
      (progn (setf res (append res (list (car even)) (list (car odd)))) (my-merge (cdr even) (cdr odd) res))
      res)
  )
|#
  
  
(even-odd '(1 2))
(even-odd '(1 10 4 13 7))
(even-odd '(1 3 5 7))
(even-odd '())
(even-odd '(2 2))
(even-odd '(9 7 5 2))