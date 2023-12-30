<?php

require_once('../server_app/nusoap.php');
// This is your Web service server WSDL URL address
$wsdl = "http://localhost:8000/index.php";

// Create client object
$client = new nusoap_client($wsdl, 'wsdl');
$err = $client->getError();
if ($err) {
   // Display the error
   echo '<h2>Constructor error</h2>' . $err;
   // At this point, you know the call that follows will fail
   exit();
}

// Call the hello method
error_log('calling function...\n');
$result1=$client->call('TransformNumber', array('x'=>'300'));
print_r($result1).'\n';

?>