(defun to-minute (h m)
	(+ (* h 60) m))

(defun time-to-meet (x)
	(* (/ 720 11) x))		

(defun count-occur-meets (h m)
	(floor (/ (to-minute h m) (time-to-meet 1))))

(defun parallel-hands-minutes (h m)
	(if (and (>= h 0)
		(< h 12)
		(>= m 0)
		(< m 60))
		(values (floor (- (time-to-meet (+ (count-occur-meets h m) 1)) (to-minute h m))))))