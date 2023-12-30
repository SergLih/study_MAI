;  half-interval.ss 2018
(define (half-interval-metod a b) ;проверяет метод половинного деления
  (let((a-value (fun a))
       (b-value (fun b))
      )
     (cond((and(< a-value 0)(> b-value 0))
                 (try a b))
         ((and(> a-value 0)(< b-value 0))
                 (try b a))
         (else(+ b 1))
     )
  ) ;готовит данные для применения вычислений
)
(define(try neg-point pos-point) ;try совпадает инициал студента
 (let( ;
       (midpoint (average neg-point pos-point)) ;ххх-заменить midpoint - середина интервала
       (test-value 0)
     ) ; test value значение функции в точке мидпоинт
     (display "+") ; при каждом вычислении печатает плюсик
     (cond((close-enough? neg-point pos-point) midpoint) ; если значения на концах интервала, то вернется
        (#t (set! test-value (fun midpoint)) ; составная ветвь
            (cond((> test-value 0)(try neg-point midpoint)) ; значение мидпоинт присваивается тест валуе
                 ((< test-value 0)(try midpoint pos-point))
                 (else midpoint))
         )
     )
 )
)
(define (close-enough? x y)
  (<(abs (- x y))tolerance))
(define (average x y)(/(+ x y)2.))
(define (root a b)
 (display"interval=\t[")
 (display a)
 (display" , ")
 (display b)
 (display"]\n")
 (let((temp (half-interval-metod a b)))
      (newline)
      (display"discrepancy=\t")
      (display(fun temp))(newline)
      (display"root=\t\t")
      (display(if(=(- temp b 1)0)"[bad]" "[good]"))
      temp 
 )
)
(define tolerance 0.00001)
(define(fun z)
  (set! z (- z (/ 108 109)(/ e)))
  
  (-(-(* z 3) (*(log z) 4)) 5)
)

" SSL variant 8"
(root 4 5)
