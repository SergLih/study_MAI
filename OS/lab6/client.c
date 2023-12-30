#include "common.h"


void menuUser() {
  printf("================================\n");
  printf("1. Add money to debit account    \n");
  printf("2. Add money to credit account   \n");
  printf("3. Withdraw money from debit account\n");
  printf("4. Withdraw money from credit account\n");
  printf("5. Get balance                       \n");  
  printf("6. Send money between debit accounts \n");
  printf("7. Send money between credit accounts\n");
  printf("0. Exit from the program             \n");  
  printf("==================================\n");
}

int main(int argc, char *argv[]) {

	TMessage message;
	int act = 0;
	char ans[256];

	void* context = zmq_ctx_new();
	void* server = zmq_socket(context, ZMQ_REQ);

	ans[0]='\0';
 	strcat(ans,"tcp://localhost:");
 	if (argc==2)
  		strcat(ans,argv[1]);
  	else {
  	    printf("Enter server's adress: ");
        scanf("%d", &act);
  		sprintf(ans, "%s%d", ans, act);
  		printf("%s\n", ans);
  	}
	int rc = zmq_connect(server, ans);
      
	if (rc != 0) {
		perror("zmq_connect");    
		zmq_close(server);
		zmq_ctx_destroy(context);    
		exit(1);
	}  
  
	printf("Enter client id:\n");
	scanf("%u", &message.clientId);
	message.amount = 0;
	message.receiverId = 0;
	message.action = R_INITIALIZE;
	zmq_msg_t clientReq;
	zmq_msg_init_size(&clientReq, sizeof(TMessage));
	memcpy(zmq_msg_data(&clientReq), &message, sizeof(TMessage));
	zmq_msg_send(&clientReq, server, ZMQ_DONTWAIT);
	zmq_msg_close(&clientReq);
	zmq_msg_t reply;
	zmq_msg_init(&reply);
	zmq_msg_recv(&reply, server, 0);
	strcpy(ans, (char*)zmq_msg_data(&reply));
  
	if (strcmp(ans, "OK") == 0) {
		printf("\nEnter user commands:\n");
	} else if (strcmp(ans, "ERROR") == 0) {
		printf("Server error.\n");
		exit(1);
	} else {
		printf("Bad server answer. Try again later.\n");
		exit(1);
	}

	zmq_msg_close(&reply);

	while(true) {
		menuUser();
		scanf("%d", &act);

		if (act == 1) {
			printf("Enter amount you want to add: ");
			scanf("%lld", &(message.amount));
			message.action = R_DEBIT_ADD;
			message.receiverId = 0;
		
		} else if (act == 2) {
			printf("Enter amount you want to add: ");
			scanf("%lld", &(message.amount));
			message.action = R_CREDIT_ADD;
			message.receiverId = 0;
		
		} else if (act == 3) {
			printf("Enter amount you want to withdraw: ");
			scanf("%lld", &(message.amount));
			message.action = R_DEBIT_WTHDRAW;
			message.receiverId = 0;
		
		} else if (act == 4) {
			printf("Enter amount you want to get: ");
			scanf("%lld", &(message.amount));
			message.action = R_CREDIT_WTHDRAW;
			message.receiverId = 0;
		
		} else if (act == 5) {
			message.amount = 0;
			message.action = R_GET_BALANCE;
			message.receiverId = 0;
		
		} else if (act == 6) {
			printf("Enter amount you want to send: ");
			scanf("%lld", &(message.amount));
			printf("Enter reciever ID: ");
			scanf("%u", &(message.receiverId));
			message.action = R_DEBIT_SEND;
		
		} else if (act == 7) {
			printf("Enter amount you want to get and send: ");
			scanf("%lld", &(message.amount));
			printf("Enter reciever ID: ");
			scanf("%u", &(message.receiverId));
			message.action = R_CREDIT_SEND;
		
		} else if (act == 0) {
			break;

		} else {
			printf("Try again...\n");
			continue;
		}

		zmq_msg_init_size(&clientReq, sizeof(TMessage));
		memcpy(zmq_msg_data(&clientReq), &message, sizeof(TMessage));

		printf("Sending...\n");
		zmq_msg_send(&clientReq, server, 0);
		zmq_msg_close(&clientReq);
		zmq_msg_init(&reply);
		zmq_msg_recv(&reply, server, 0);

		strcpy(ans, (char*)zmq_msg_data(&reply));
		if (strcmp(ans, "ERROR") == 0) {
	  		printf("Error occured. \n");
		} else {
	 		printf("%s\n",ans);
		}
		zmq_msg_close(&reply);

  	}
  zmq_close(server);
  zmq_ctx_destroy(context);
  return 0;
}
