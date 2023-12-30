using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WS0
{
	class Program
	{
	public static void Main (string[] arg) {
	    if(arg.Length != 1) {
			Console.WriteLine("Usage: lab8.exe <input_string>");
			return;
		}
		var serv = new serv_wsdlphp();
		string res = serv.prog4(arg[0]);       // 1 параметр
		Console.WriteLine("prog4 : " + res);   // вывод
	}
	}
}
