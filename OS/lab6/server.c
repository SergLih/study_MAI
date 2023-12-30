#include "vector.h"
#include <zmq.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

int main(int argc, char *argv[]) {
    TVector* vector = VectorCreate(10);
    TMessage* message;
    zmq_msg_t reply;
    zmq_msg_t request;
    char ans[256];
    void* context = zmq_ctx_new();
    void* responder = zmq_socket(context, ZMQ_REP);
    ans[0]='\0';
    strcat(ans,"tcp://*:");
    if(argc==2)
  	    strcat(ans,argv[1]);
    else
  	    strcat(ans,"4040");
    int rc = zmq_bind(responder, ans);

    if (rc != 0) {
        perror("zmq_bind");	
        zmq_close(responder);
        zmq_ctx_destroy(context); 
        exit(1);
    }
  

    printf("Server initialized\n");
    while(true) {
        zmq_msg_init(&request);
        zmq_msg_recv(&request, responder, 0);
        message = (TMessage*) zmq_msg_data(&request);
        printf("Recieved message from %u action: %d \n", message->clientId, message->action);

        if (VectorSearch(vector, message->clientId)<0) {
            //добавить нового клиента
            printf("Client %u added successfully\n",message->clientId);
            VectorAdd(vector, message->clientId);
	    }      
	
	    if (R_INITIALIZE == message->action) {
		    if(VectorSearch(vector, message->clientId)>=0)
		    	sprintf(ans, "OK");
		    else
			    sprintf(ans, "ERROR");

	    } else if (R_DEBIT_ADD == message->action) {
		    if(AddMoney(vector, message->amount, message->clientId, A_DEBIT))
			    sprintf(ans, "OK");
		    else
			    sprintf(ans, "ERROR");

	    } else if (R_CREDIT_ADD == message->action) {
		    if(AddMoney(vector, message->amount, message->clientId, A_CREDIT))
		    	sprintf(ans, "OK");
		    else
		    	sprintf(ans, "ERROR");

	    } else if (R_DEBIT_WTHDRAW == message->action) {
		    if (WithdrawMoney(vector, message->amount, message->clientId, A_DEBIT))
		    	sprintf(ans, "OK");
		    else
		    	sprintf(ans, "Error: not enough money on debit and/or credit account");
	
        } else if (R_CREDIT_WTHDRAW == message->action) {
		    if(WithdrawMoney(vector, message->amount, message->clientId, A_CREDIT))
			    sprintf(ans, "OK");
		    else
			    sprintf(ans, "Error: not enough money on credit account");
	
	    } else if (R_GET_BALANCE == message->action) {
	    	if(!GetBalance(vector, message->clientId, ans))
	    		sprintf(ans, "ERROR");

	    } else if (R_DEBIT_SEND == message->action) {
		    if(VectorSearch(vector, message->receiverId)<0)
			    sprintf(ans, "No such receiver exists");
		    else {
			    if (WithdrawMoney(vector, message->amount, message->clientId, A_DEBIT)) {
				    if (!AddMoney(vector, message->amount, message->receiverId, A_DEBIT)) {
					    sprintf(ans, "ERROR");
					    //если невозможно пополнить счет получателя, то пытаемся вернуть отправителю деньги
					    //если не получается это сделать, то фатальная ошибка
					    if (!AddMoney(vector, message->amount, message->clientId, A_DEBIT)) {
						    printf("FATAL ERROR: cannot roll back transaction\n ");
						    printf("\tcan not transfer back %lld money to %u\n ", 
							    message->amount, message->clientId);
						    break;
					    }
				    }
				    sprintf(ans, "OK");
			    } else
				    sprintf(ans, "Error: not enough money on debit and/or credit account");
		    }
	    } else if (R_CREDIT_SEND == message->action) {
		    if(VectorSearch(vector, message->receiverId) < 0)
			    sprintf(ans, "No such receiver exists");
		    else {
			    if (WithdrawMoney(vector, message->amount, message->clientId, A_CREDIT)) {
				    if (!AddMoney(vector, message->amount, message->receiverId, A_CREDIT)) {
                        sprintf(ans, "ERROR");
                        //если невозможно пополнить счет получателя, то пытаемся вернуть отправителю деньги
                        //если не получается это сделать, то фатальная ошибка
                        if (!AddMoney(vector, message->amount, message->clientId, A_CREDIT)) {
                            printf("FATAL ERROR: cannot roll back transaction\n ");
                            break;
                        }
				    }
				    sprintf(ans, "OK");
			    } else
                    sprintf(ans, "Error: not enough money on credit account");	
            }

        } else {
            sprintf(ans, "ERROR: Wrong request");
        }

        printf("Send answer to client: [%s]\n", ans);
        zmq_msg_close(&request);
        zmq_msg_init_size(&reply, strlen(ans)+1);
        memcpy(zmq_msg_data(&reply), ans, strlen(ans)+1);
        zmq_msg_send(&reply, responder, 0);
        zmq_msg_close(&reply);
    }
  
  zmq_close(responder);
  zmq_ctx_destroy(context); 
  VectorDestroy(vector); 
  return 0;
}

