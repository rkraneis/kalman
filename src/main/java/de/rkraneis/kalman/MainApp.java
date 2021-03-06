package de.rkraneis.kalman;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.filter.DefaultMeasurementModel;
import org.apache.commons.math3.filter.DefaultProcessModel;
import org.apache.commons.math3.filter.KalmanFilter;
import org.apache.commons.math3.filter.MeasurementModel;
import org.apache.commons.math3.filter.ProcessModel;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixDimensionMismatchException;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

public class MainApp extends Application {

    @Override
    public void start(Stage primaryStage) {
        Group root = new Group();
        primaryStage.setScene(new Scene(root));

        List<XYChart.Data<Double, Double>> s0 = new ArrayList<>();
        List<XYChart.Data<Double, Double>> s1 = new ArrayList<>();
        List<XYChart.Data<Double, Double>> s2 = new ArrayList<>();

        kalman2(s0, s1, s2);

        LineChart chart = createChart(s0, s1, s2);
        root.getChildren().add(chart);
        primaryStage.show();
    }

    private void kalman1(List<XYChart.Data<Double, Double>> l0, List<XYChart.Data<Double, Double>> l1) throws MatrixDimensionMismatchException, NoDataException, DimensionMismatchException, NullArgumentException, SingularMatrixException, OutOfRangeException {
        // increasing speed example
        // v=a*t

        // discrete time interval
        double dt = 0.1d;
        // position measurement noise (meter)
        double measurementNoise = 10d;
        // acceleration noise (meter/sec^2)
        double accelNoise = 0.2d;

        // state transition matrix
        // A = [ 1  dt ]
        //     [ 0  1  ]
        RealMatrix A = matrix(
                $(1, dt),
                $(0, 1)
        );

        // control input matrix
        // B = [ dt^2/2 ]
        //     [ dt     ]
        RealMatrix B = matrix(
                $(Math.pow(dt, 2d) / 2d),
                $(dt)
        );

        // measurement matrix
        // H = [ 1 0 ]
        RealMatrix H = matrix(
                $(1d, 0d)
        );

        // x = [ 0 0 ]
        RealVector x = vector(
                0, 0
        );

        // process noise covariance matrix
        // Q = [ dt^4/4 dt^3/2 ]
        //     [ dt^3/2 dt^2   ]
        RealMatrix tmp = matrix(
                $(Math.pow(dt, 4d) / 4d, Math.pow(dt, 3d) / 2d),
                $(Math.pow(dt, 3d) / 2d, Math.pow(dt, 2d))
        );
        RealMatrix Q = tmp.scalarMultiply(Math.pow(accelNoise, 2));

        // error covariance matrix
        // P0 = [ 1 1 ]
        //      [ 1 1 ]
        RealMatrix P0 = matrix(
                $(1, 1),
                $(1, 1)
        );

        // measurement noise covariance matrix
        // R = [ measurementNoise^2 ]
        RealMatrix R = new Array2DRowRealMatrix(new double[]{
            Math.pow(measurementNoise, 2)
        });

        // constant control input, increase velocity by 0.1 m/s per cycle
        RealVector u = vector(0.1d);

        ProcessModel pm = new DefaultProcessModel(A, B, Q, x, P0);
        MeasurementModel mm = new DefaultMeasurementModel(H, R);
        KalmanFilter filter = new KalmanFilter(pm, mm);

        RandomGenerator rand = new JDKRandomGenerator();

        RealVector tmpPNoise = vector(
                Math.pow(dt, 2d) / 2d,
                dt
        );
        RealVector mNoise = new ArrayRealVector(1);

        double position = filter.getStateEstimation()[0];
        double velocity = filter.getStateEstimation()[1];
        l0.add(new XYChart.Data<>(position, velocity));
        double realPosition = x.getEntry(0);
        double realVelocity = x.getEntry(1);
        l1.add(new XYChart.Data<>(realPosition, realVelocity));

        // iterate 60 steps
        for (int i = 0; i < 60; i++) {
            filter.predict(u);

            // simulate the process
            RealVector pNoise = tmpPNoise.mapMultiply(accelNoise * rand.nextGaussian());

            // x = A * x + B * u + pNoise
            x = A.operate(x).add(B.operate(u)).add(pNoise);

            realPosition = x.getEntry(0);
            realVelocity = x.getEntry(1);
            l1.add(new XYChart.Data<>(realPosition, realVelocity));

            // simulate the measurement
            mNoise.setEntry(0, measurementNoise * rand.nextGaussian());

            // z = H * x + m_noise
            RealVector z = H.operate(x).add(mNoise);

            filter.correct(z);

            position = filter.getStateEstimation()[0];
            velocity = filter.getStateEstimation()[1];
            l0.add(new XYChart.Data<>(position, velocity));
        }
    }

    private void kalman2(List<XYChart.Data<Double, Double>> l0, List<XYChart.Data<Double, Double>> l1, List<XYChart.Data<Double, Double>> l2) throws MatrixDimensionMismatchException, NoDataException, DimensionMismatchException, NullArgumentException, SingularMatrixException, OutOfRangeException {
        // cannonball
        // x(t)   = x_0 + v_0x*t
        // v_x(t) = v_0x
        // y(t)   = y_0 + v_0y*t - 1/2*g*t*t
        // v_y(t) = v_0y - g*t

        // discrete time interval
        double dt = 0.1;

        // position measurement noise (meter)
        double measurementNoise = 30;

        double v_abs = 100;
        double angle = Math.toRadians(45);

        double v_x = v_abs * Math.cos(angle);
        double v_y = v_abs * Math.sin(angle);

        double g = 9.81;

        // state transition matrix
        //     [ 1  dt 0 0  ]
        // A = [ 0  1  0 0  ]
        //     [ 0  0  1 dt ]
        //     [ 0  0  0 1  ]
        RealMatrix A = matrix(
                $(1, dt, 0, 0),
                $(0, 1, 0, 0),
                $(0, 0, 1, dt),
                $(0, 0, 0, 1)
        );

        // control input matrix
        RealMatrix B = diag(0, 0, 1, 1);

        // control input vector
        RealVector u = vector(0, 0, 0.5 * -g * dt * dt, -g * dt);

        // measurement matrix; only x and y
        RealMatrix H = diag(1, 0, 1, 0);

        // initial state
        // x = [ 0  vx  500  vy  ]
        RealVector x = vector(0, v_x, 500, v_y);

        // error covariance matrix
        // P0 = I
        RealMatrix P0 = id(4);

        // process noise covariance matrix
        // Q = 0
        RealMatrix Q = zero(4, 4);

        // measurement noise covariance matrix
        // R = I * 0.2
        RealMatrix R = id(4).scalarMultiply(0.2);

        ProcessModel pm = new DefaultProcessModel(A, B, Q, x, P0);
        MeasurementModel mm = new DefaultMeasurementModel(H, R);
        KalmanFilter filter = new KalmanFilter(pm, mm);

        RealVector x2 = vector(0, v_x, 0, v_y);

        RandomGenerator rnd = new JDKRandomGenerator();

        // iterate 144 steps
        for (int i = 0; i < 144; i++) {
            System.out.println(i);

            l2.add(data(x2.getEntry(0), x2.getEntry(2)));

            RealVector x1 = x2.mapAdd(measurementNoise * rnd.nextGaussian());
            l1.add(data(x1.getEntry(0), x1.getEntry(2)));

            // iterate cannon
            x2 = A.operate(x2).add(B.operate(u));

            double[] x0 = filter.getStateEstimation();
            l0.add(data(x0[0], x0[2]));

            filter.predict(u);
            filter.correct(x1);
        }
    }

    private <T> LineChart createChart(List<XYChart.Data<T, T>>... ls) {
        List<XYChart.Series<T, T>> series = new ArrayList<>(ls.length);
        for (int i = 0; i < ls.length; i++) {
            List<XYChart.Data<T, T>> l = ls[i];
            series.add(new LineChart.Series<>(Integer.toString(i),
                    FXCollections.observableArrayList(l)));
        }
        ObservableList<XYChart.Series<T, T>> lineChartData
                = FXCollections.observableArrayList(series);
        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("X");
        yAxis.setLabel("Y");
        LineChart chart = new LineChart(xAxis, yAxis, lineChartData);
        chart.setCreateSymbols(false);
        return chart;
    }

    private void fillInData(List<XYChart.Data<Double, Double>> l1, List<XYChart.Data<Double, Double>> l2) {
        Random rnd = new Random();
        for (int i = 0; i <= 360; i += 10) {
            double angle_rad = Math.toRadians(i);
            double x = Math.cos(angle_rad);
            double y = Math.sin(angle_rad);
            l1.add(new XYChart.Data<>(x, y));
            double x_pert = x + rnd.nextGaussian() * 0.1;
            double y_pert = y + rnd.nextGaussian() * 0.1;
            l2.add(new XYChart.Data<>(x_pert, y_pert));
        }
    }

    /**
     * The main() method is ignored in correctly deployed JavaFX application.
     * main() serves only as fallback in case the application can not be
     * launched through deployment artifacts, e.g., in IDEs with limited FX
     * support. NetBeans ignores main().
     *
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        launch(args);
    }

    private static RealVector vector(double... ds) {
        return MatrixUtils.createRealVector(ds);
    }

    private static double[] $(double... ds) {
        return ds;
    }

    private static RealMatrix matrix(double[]... rows) {
        return MatrixUtils.createRealMatrix(rows);
    }

    private static RealMatrix matrix(RealVector... cols) {
        int numCols = cols.length;
        int numRows = cols[0].getDimension();
        RealMatrix mat = MatrixUtils.createRealMatrix(numRows, numCols);
        for (int numCol = 0; numCol < numCols; numCol++) {
            mat.setColumnVector(numCol, cols[numCol]);
        }
        return mat;
    }

    private static RealMatrix diag(double... ds) {
        return MatrixUtils.createRealDiagonalMatrix(ds);
    }

    private static RealMatrix id(int d) {
        return MatrixUtils.createRealIdentityMatrix(d);
    }

    private static RealMatrix zero(int m, int n) {
        return MatrixUtils.createRealMatrix(m, n);
    }

    private static <T> XYChart.Data<T, T> data(T x, T y) {
        return new XYChart.Data<>(x, y);
    }
}
