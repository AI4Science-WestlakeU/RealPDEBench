import java.util.*;
import org.apache.xmlrpc.*;
import java.util.concurrent.*;
import java.util.concurrent.Semaphore;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.JSONArray;
Test0 test;  
SaveScalar dat;
SaveVectorField data;
PrintWriter output1;

int EpisodeTime = 15500; 
int Re = 3272;
int resolution = 48, xLengths = 8, yLengths = 8, zoom = 100/resolution;
int plotTime = 0;
int picNum = 10;
int simNum = 1;
float tStep = .1;
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
float St=.45;
Semaphore state_sem = new Semaphore(0);

float theta0lowerBound = 0.0; 
float theta0upperBound = (PI*resolution/2)*(tStep); 
int startXlowerBound = 84, startYlowerBound = 33, targetXlowerBound = 143, targetYlowerBound = 33; 
int startXupperBound = 87, startYupperBound = 77, targetXupperBound = 146, targetYupperBound = 77; 
float target_range = 2;
PVector t_loca = new PVector(), target = new PVector();
PVector xc_1= new PVector(), xc_2= new PVector(), xc_3= new PVector(), xc_4= new PVector();
Random random = new Random();
int py = 383;  
int px = 0; 
int py_i = 0;  
int px_e = 383; 
int action_denm = 1; 
int iter = 0; 
float[] theta = {0.0,0.0,0.0,0.0};
String[][] velocity_u = new String[384][384];
String[][] velocity_v = new String[384][384];
String[][] pressure_p = new String[384][384];
String[][] bd_x = new String[2][40];
String[][] bd_y = new String[2][40];
float CD_0=0.0, CL_0=0.0, CD_1=0.0, CL_1=0.0;
float diameter = 1.0;
float gap = 2.0, dia_0 = 1, dia_1 = 1;
float fn_C=0.01;
float epi_C=0.8;  //cr/c
float massR_C=15;//mass ratio

void settings()
{
  size(600,600);
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
  setUpNewSim(EpisodeNum, Re, epi_C, massR_C);
}

void draw(){
  if (test.t < EpisodeTime){
    
    if (test.t>initTime){
      callLearn--;
      picNum--;

      NextAction = callActionByRemote();

      test.update();

      CD_0 = test.force_0.x*2/resolution;
      CL_0 = test.force_0.y*2/resolution;
      CD_1 = test.force_1.x*2/resolution;
      CL_1 = test.force_1.y*2/resolution;

      if (callLearn<=0){
        callLearn = 10;
      }
      
      if(test.t>plotTime){
        if(picNum <= 0){
        test.display();
        // If want to save figure plot:
        saveFrame("saved/"+str(simNum) + "/" +"frame-#######.png");
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

    String force_0 = multy_state_force(CD_0, CD_1);
    String force_1 = multy_state_force(CL_0, CL_1);
    String angle_all = multy_state_angle(test.body.bodyList.get(0).phi, test.body.bodyList.get(1).phi);
    String state_u = arrayToJSON(velocity_u);
    String state_v = arrayToJSON(velocity_v);
    String state_p = arrayToJSON(pressure_p);

    for (int k=0; k < test.body.bodyList.size(); k++)  {
      int i = 0;
      for (PVector vec : test.body.bodyList.get(k).coords){
        bd_x[k][i] = String.valueOf(vec.x);
        i += 1;
      }
    }

    for (int k=0; k < test.body.bodyList.size(); k++)  {
      int i = 0;
      for (PVector vec : test.body.bodyList.get(k).coords){
        bd_y[k][i] = String.valueOf(vec.y);
        i += 1;
      }
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
    Re = 3000;
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

        if(test.t==0.1){
          Re = int(input_object.getFloat("Re"));
          epi_C = input_object.getFloat("epi_C");
          massR_C = input_object.getFloat("massR_C");
          setUpNewSim(EpisodeNum, Re, epi_C, massR_C);
        }
        
        action[0] = input_object.getFloat("v1");
        action[1] = input_object.getFloat("v2");

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

public String multy_state_force(float C0, float C1) {
  JSONObject multy_state_json = new JSONObject();
  multy_state_json.put("C0", C0);
  multy_state_json.put("C1", C1);
 return multy_state_json.toJSONString();
}

public String multy_state_angle(float an_1, float an_2) {
  JSONObject multy_state_json = new JSONObject();
  multy_state_json.put("0", an_1);
  multy_state_json.put("1", an_2);
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

void setUpNewSim(int runNum, int Re, float epi_C, float massR_C){       
  int xLengths = 8, yLengths = 8, zoom = 100/resolution;      
  float xi0 = 0, xi1 = 0;
  smooth(); 

  if (zoom <= 1){zoom = 1;}

  float radius=resolution; // R of the cylinder

  xc_1.x=1.5*resolution; xc_1.y=4*resolution;
  test = new Test0(resolution, radius, massR_C, fn_C, epi_C, Re, St, true); 

  callLearn = 10;
  picNum = 10;
  test.t = 0;

  test.update();

  new File(datapath + str(runNum)).mkdir();

  for (int j=round(py); j>=py_i; j--) { 
      for (int i=round(px); i<=px_e; i++) { 
        velocity_u[i][j] = String.valueOf(test.flow.u.x.a[i][j]);
        velocity_v[i][j] = String.valueOf(test.flow.u.y.a[i][j]);
        pressure_p[i][j] = String.valueOf(test.flow.p.linear(i,j));
      }
      
    }

  String force_0 = multy_state_force(CD_0, CD_1);
  String force_1 = multy_state_force(CL_0, CL_1);
  String angle_all = multy_state_angle(test.body.bodyList.get(0).phi, test.body.bodyList.get(1).phi);
  String state_u = arrayToJSON(velocity_u);
  String state_v = arrayToJSON(velocity_v);
  String state_p = arrayToJSON(pressure_p);

  for (int k=0; k < test.body.bodyList.size(); k++)  {
    int i = 0;
    for (PVector vec : test.body.bodyList.get(k).coords){
      bd_x[k][i] = String.valueOf(vec.x);
      i += 1;
    }
  }

  for (int k=0; k < test.body.bodyList.size(); k++)  {
    int i = 0;
    for (PVector vec : test.body.bodyList.get(k).coords){
      bd_y[k][i] = String.valueOf(vec.y);
      i += 1;
    }
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
