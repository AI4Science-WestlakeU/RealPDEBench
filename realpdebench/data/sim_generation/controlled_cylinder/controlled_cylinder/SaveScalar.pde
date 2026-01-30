/**********************************
 SaveScalar class
 
 Saves scalar data to a text file with customizable header
 
 example code:
 // dat = new SaveData("pressure.txt",test.body.);
 dat.addData(test.t, test.flow.p);
 dat.finish();
 ***********************************/

class SaveScalar{
  PVector force;
  PrintWriter output;
  Pinball test;  
  int n;
  float r;
  float  resolution, xi1, xi2, theta, D;
  float centX_f, centY_f, centX_b, centY_b, centX_t, centY_t;
  int numTheta;
  
  SaveScalar(String name){
    output = createWriter(name);
    output.println("%% Force coefficients using processing viscous simulation");
    output.println();
    output.println("%% Fellowing: t, force.x, force.y");
    output.println();
  }
  
  SaveScalar(String name, float res, float xLen, float yLen, int num){
    output = createWriter(name);
    output.println("%% Force and pressure coefficients using processing viscous simulation");
    output.println();
    output.println("%% Fellowing: t, force.x, force.y");
    output.println();
    this.resolution = res;
    this.D = res;
    float n = xLen*res;
    float m = yLen*res;
    //r = test.r;
    //theta = test.theta;
    //this.numTheta = num;
    //this.centX_f = n/4;
    //this.centY_f = m/2;
    //this.centX_b = n/4+r*cos(theta);
    //this.centY_b = m/2-r*sin(theta);
    //this.centX_t = n/4+r*cos(theta);
    //this.centY_t = m/2+r*sin(theta);
  }
     
  void addData(float t,PVector force){
    output.println(t + " " + force.x + " " +force.y);
    output.println("");
  }
  
  void addData02(float t,PVector force, Field pres){
    output.print(t + " " + force.x + " " +force.y + " ");
    //for(int i=0;i<numTheta; i++){
    //  float xPre = cos((float)i/numTheta*PI*2)*D/2 + centX;
    //  float yPre = sin((float)i/numTheta*PI*2)*D/2 + centY;
      
    //  float pdl = pres.linear( xPre, yPre );
    //  output.print(pdl + " ");
    //}
    //output.println("");
    output.println("");
  }
  
  void addData03(float t, float r0, float r1, float r2, PVector force, PVector force_1, PVector force_2, Field pres, VectorField uvel){
    output.print(t + " " + r0 + " " + r1 + " " + r2 + " " + force.x + " " +force.y + " " + force_1.x + " " +force_1.y + " " 
                 + force_2.x + " " +force_2.y + " " );
    //for(int i=0;i<numTheta; i++){   //0-15
    //  float xPre_f = cos((float)i/numTheta*PI*2)*D/2 + centX_f;
    //  float yPre_f = sin((float)i/numTheta*PI*2)*D/2 + centY_f;
    //  float xPre_b = cos((float)i/numTheta*PI*2)*D/2 + centX_b;
    //  float yPre_b = sin((float)i/numTheta*PI*2)*D/2 + centY_b;
    //  float xPre_t = cos((float)i/numTheta*PI*2)*D/2 + centX_t;
    //  float yPre_t = sin((float)i/numTheta*PI*2)*D/2 + centY_t;
      
    //  float pdl_f = pres.linear( xPre_f, yPre_f );
    //  float pdl_b = pres.linear( xPre_b, yPre_b );
    //  float pdl_t = pres.linear( xPre_t, yPre_t );
    //  //output.print(pdl + " ");
    //}
    output.println("");
  }
  
  
  void finish(){
    output.flush(); // Writes the remaining data to the file
    output.close(); // Finishes the file
  }
} 
  
