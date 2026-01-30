import java.util.*;
import org.apache.xmlrpc.*;
import java.util.concurrent.*;
import java.util.concurrent.Semaphore;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.JSONArray;
Single_Cylinder test;  
SaveScalar dat;
SaveVectorField data;
PrintWriter output1;

int EpisodeTime = 30; 
int Re = 100;
int resolution = 16, xLengths = 16, yLengths = 8, zoom = 100/resolution;
int plotTime = 20;
int picNum = 10;
int simNum = 1;
float tStep = .01;
String datapath = "saved" + "/";
int EpisodeNum = 1;
int initTime = 0, callLearn = 10;
float y = 0, angle = 0;
float cost = 0;
String p;
XmlRpcClient client;
WebServer server;
float nextactionD=0, nextactionC=0, nextactionB=0,nextactionA=0;
float[] NextAction = {nextactionA,nextactionB,nextactionC,nextactionD};
ArrayList state = new ArrayList();
ArrayList boundary = new ArrayList();
float[] action = new float[2];
Boolean done = false;
Semaphore action_sem = new Semaphore(0);
Semaphore state_sem = new Semaphore(0);

float theta0lowerBound = 0.0; 
float theta0upperBound = (PI*resolution/2)*(tStep); 
int startXlowerBound = 84, startYlowerBound = 33, targetXlowerBound = 143, targetYlowerBound = 33; 
int startXupperBound = 87, startYupperBound = 77, targetXupperBound = 146, targetYupperBound = 77; 
float target_range = 2;
PVector t_loca = new PVector(), target = new PVector();
PVector xc_1= new PVector(), xc_2= new PVector(), xc_3= new PVector(), xc_4= new PVector();
Random random = new Random();
int py = 127;  
int px = 0; 
int py_i = 0;  
int px_e = 127; 
int action_denm = 1; 
int iter = 0; 
float[] theta = {0.0,0.0,0.0,0.0};
String[][] velocity_u = new String[128][128];
String[][] velocity_v = new String[128][128];
String[][] pressure_p = new String[128][128];
String[][] bd_x = new String[1][40];
String[][] bd_y = new String[1][40];
float CD_0=0.0, CL_0=0.0, CD_1=0.0, CL_1=0.0;
float diameter = 1.0;
float gap = 2.0, dia_0 = 1, dia_1 = 1;

void settings()
{
  size(8*resolution, 8*resolution);
}

void setup()
{
  try
  {
    server = StartServer(int(args[0])); 
  }
  catch (Exception ex)
  {
    println(ex);
  }
  setUpNewSim(EpisodeNum, Re, dia_0);
}

void draw(){
  if (test.t < EpisodeTime){
    
    if (test.t>initTime){
      callLearn--;
      picNum--;

      NextAction = callActionByRemote();
      test.xi0 = NextAction[0];
      test.xi0_m = action_denm*test.xi0;

      test.update2();

      CD_0 = test.force_0.x*2/resolution;
      CL_0 = test.force_0.y*2/resolution;

      if (callLearn<=0){
        callLearn = 10;
      }
      
      if(test.t>plotTime){
        if(picNum <= 0){
        test.display(t_loca.x,t_loca.y,target.x,target.y);
        picNum = 10;
        }
      }
      
    }

    for (int j=round(py); j>=py_i; j--) {
      for (int i=round(px); i<=px_e; i++) { 
        velocity_u[i][j] = String.valueOf(test.flow.u.x.a[i][j]);
        velocity_v[i][j] = String.valueOf(test.flow.u.y.a[i][j]);
        pressure_p[i][j] = String.valueOf(test.flow.p.linear(i,j));
      }
    }

    String force_0 = multy_state_force(CD_0);
    String force_1 = multy_state_force(CL_0);
    String angle_all = multy_state_angle(test.body.phi);
    String state_u = arrayToJSON(velocity_u);
    String state_v = arrayToJSON(velocity_v);
    String state_p = arrayToJSON(pressure_p);

    int i = 0;
    for (PVector vec : test.body.coords){
      bd_x[0][i] = String.valueOf(vec.x);
      i += 1;
    }
    
    i = 0;
    for (PVector vec : test.body.coords){
      bd_y[0][i] = String.valueOf(vec.y);
      i += 1;
    }
    
    
    String boundary_x = arrayToJSON(bd_x);
    String boundary_y = arrayToJSON(bd_y);
    
    state.clear();
    state.add(state_u);
    state.add(state_v);
    state.add(state_p);
    state.add(angle_all);
    state.add(force_0);
    state.add(force_1);

    boundary.clear();
    boundary.add(boundary_x);
    boundary.add(boundary_y);

    release_state();
  }     
}

float[] callActionByRemote()
{
  try
  {
    action_sem.acquire();
  }
  catch (Exception ex)
  {
    System.out.println(ex);
  }
  return action;
}

void release_state()
{
  
  if ((test.t + 0.01) > EpisodeTime){
    done = true;
    state_sem.release();
    simNum = simNum + 1;
    Re = 100;
  }
  state_sem.release();
}

WebServer StartServer(int port)
{
  println(port);
  WebServer server = new WebServer(port);
  server.addHandler("connect", new serverHandler());
  server.start();

  System.out.println("Started server successfully.");
  System.out.println("Accepting requests. (Halt program to stop.)");
  return server;
}

public
class serverHandler
{
    public String Step(String actionInJson)
      {
        JSONObject input_object = JSONObject.parseObject(actionInJson);
        JSONObject output_object = new JSONObject();

        if(test.t==0.01){
          Re = int(input_object.getFloat("Re"));
          dia_0 = input_object.getFloat("dia_0");
          setUpNewSim(EpisodeNum, Re, dia_0);
        }
        
        action[0] = input_object.getFloat("v1");

        action_sem.release();

        try {
            state_sem.acquire();
        } catch (InterruptedException e) {
            println(e);
            println("[Error] state do not refresh");
        } finally {
        }
        output_object.put("done", done);
        output_object.put("state", state);
        output_object.put("bdy", boundary);

        return output_object.toJSONString();
      }

    public String reset(String actionInJson)
      {
        JSONObject input_object = JSONObject.parseObject(actionInJson);
        JSONObject output_object = new JSONObject();

        done = false;
        action[0] = input_object.getFloat("v1");
        action[1] = input_object.getFloat("v2");

        output_object.put("state", state);
        output_object.put("done", done);
        output_object.put("bdy", boundary);
        println("Complete Reset");
        return output_object.toJSONString();
      }
}

public String multy_state_force(float C0) {
  JSONObject multy_state_json = new JSONObject();
  multy_state_json.put("C0", C0);
 return multy_state_json.toJSONString();
}

public String multy_state_angle(float an_1) {
  JSONObject multy_state_json = new JSONObject();
  multy_state_json.put("0", an_1);
 return multy_state_json.toJSONString();
}

public String arrayToJSON(String[][] uvp) {
    JSONArray jsonArray = new JSONArray();    
    for (int i = 0; i < uvp.length; i++) {
        for (int j = 0; j < uvp[i].length; j++) {
            String parts = uvp[i][j];
            float cell = Float.parseFloat(parts);
            
            JSONObject jsonObject = new JSONObject();
            jsonObject.put(i+","+j, cell);
            jsonArray.add(jsonObject);
        }
    }
    return jsonArray.toJSONString();
}

void setUpNewSim(int runNum, int Re, float dia_0){       
  int xLengths = 8, yLengths = 8, zoom = 100/resolution;      
  float xi0 = 0, xi1 = 0;
  smooth(); 

  if (zoom <= 1){zoom = 1;}

  xc_1.x=1.5*resolution; xc_1.y=4*resolution;
  test = new Single_Cylinder(resolution, Re, xc_1, dia_0, xi0, tStep, xLengths, yLengths, false);        

  callLearn = 10;
  picNum = 10;
  test.t = 0;

  test.xi0_m = theta[0];
  test.update2();

  new File(datapath + str(runNum)).mkdir();

  for (int j=round(py); j>=py_i; j--) { 
      for (int i=round(px); i<=px_e; i++) { 
        velocity_u[i][j] = String.valueOf(test.flow.u.x.a[i][j]);
        velocity_v[i][j] = String.valueOf(test.flow.u.y.a[i][j]);
        pressure_p[i][j] = String.valueOf(test.flow.p.linear(i,j));
      }
      
    }

  String force_0 = multy_state_force(CD_0);
  String force_1 = multy_state_force(CL_0);
  String angle_all = multy_state_angle(test.body.phi);
  String state_u = arrayToJSON(velocity_u);
  String state_v = arrayToJSON(velocity_v);
  String state_p = arrayToJSON(pressure_p);

  int i = 0;
  for (PVector vec : test.body.coords){
    bd_x[0][i] = String.valueOf(vec.x);
    i += 1;
  }
  
  i = 0;
  for (PVector vec : test.body.coords){
    bd_y[0][i] = String.valueOf(vec.y);
    i += 1;
  }
  
  String boundary_x = arrayToJSON(bd_x);
  String boundary_y = arrayToJSON(bd_y);
  
  state.clear();
  state.add(state_u);
  state.add(state_v);
  state.add(state_p);
  state.add(angle_all);
  state.add(force_0);
  state.add(force_1);

  boundary.clear();
  boundary.add(boundary_x);
  boundary.add(boundary_y);
        
  println("Episode");
}
