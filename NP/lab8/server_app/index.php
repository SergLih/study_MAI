<?php
require_once('nusoap.php');  // упрощает создание клиентов и серверов SOAP
$server = new soap_server(); // Создаем экземпляр сервера
$server->configureWSDL('serv_wsdl.php', 'urn:serv_wsdl'); 
// Устанавливаем пространство имен с префиксом tns для WSDL-схемы
$server->wsdl->schemaTargetNamespace = 'urn:serv_wsdl';
error_reporting(E_ALL);
ini_set('display_errors', 'On');

function TransformNumber($x) {  // Определяем метод как функцию PHP
	error_log('function has been called...\n');
	$x = floatval($x);
	if($x > 0)
		$res = $x - 2.0;
	elseif($x == 0)
		$res = 0.0;
	else
		$res = $x*3;
	return "prog4: $res";
}
$server->register('TransformNumber', 				// Регистрация метода
	array('x' => 'xsd:string'), 		// входные параметры
	array('return' => 'xsd:string'), 	// выходные параметры
		'urn:serv_wsdl', 	'urn:serv_wsdl#TransformNumber', 
		'rpc',  'encoded', 'TransformNumber x' );   //   стиль,   описание

// При запуске index.php получаем wsdl
// Используем HTTP-запрос чтобы вызвать сервис
	$par = file_get_contents("php://input");
	$server->service($par); 
; 
?>
