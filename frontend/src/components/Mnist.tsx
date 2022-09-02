import * as React from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';

import './Mnist.css';
import { ensureExists } from 'src/utils';
const MNIST_WIDTH_PX = 28;
const LINE_WIDTH = 2.5;

interface ModelResult {
  rank: number;
  number: number;
  generation: number;
}

let generation = 0;
export function Mnist() {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const selectRef = React.useRef<HTMLSelectElement>(null);
  const ctxRef = React.useRef<CanvasRenderingContext2D | null>(null);
  const [curve, setCurve] = React.useState<Curve | null>(null);
  const [model, setModel] = React.useState<tf.LayersModel | null>(null);
  const [modelPath, setModelPath] = React.useState<string | null>(null);
  const [modelResults, setModelResults] = React.useState<ModelResult[] | null>(
    null,
  );

  React.useEffect(() => {
    const drawingTarget = canvasRef.current;
    if (!drawingTarget) {
      return;
    }
    const ctx = drawingTarget.getContext('2d');
    if (!ctx) {
      return;
    }
    ctxRef.current = ctx;
    changeModel();
    setupCurveDrawing({
      drawingTarget,
      onStartDraw() {
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, MNIST_WIDTH_PX, MNIST_WIDTH_PX);
      },
      onCurveDrawn(curve: Curve) {
        const s =
          drawingTarget.getBoundingClientRect().width / drawingTarget.width;
        function scale(points: Vec2[]) {
          for (const point of points) {
            point.x /= s;
            point.y /= s;
          }
        }
        console.log(curve.line[0]);
        scale(curve.line);
        scale(curve.cpLeft);
        scale(curve.cpRight);
        setCurve(curve);
        console.log(curve.line[0]);
      },
      onPoint(prev: Vec2, curr: Vec2) {
        const s =
          drawingTarget.getBoundingClientRect().width / drawingTarget.width;

        ctx.lineWidth = LINE_WIDTH;
        ctx.strokeStyle = '#000';
        ctx.lineCap = 'round';

        ctx.beginPath();
        ctx.moveTo(prev.x / s, prev.y / s);
        ctx.lineTo(curr.x / s, curr.y / s);
        ctx.stroke();
      },
      pointsPerDistance: 5,
    });
  }, []);

  React.useEffect(() => {
    if (modelPath) {
      tf.setBackend('wasm').then(() => {
        tf.loadLayersModel(modelPath).then((model) => {
          console.log('Setting model', model);
          setModel(model);
        });
      });
    }
  }, [modelPath]);

  React.useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !curve || !model) {
      return;
    }
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, MNIST_WIDTH_PX, MNIST_WIDTH_PX);
    drawCurve(ctx, curve);
    drawCurve(ctx, curve);
    const { data } = ctx.getImageData(0, 0, MNIST_WIDTH_PX, MNIST_WIDTH_PX);
    const tensorData = new Float32Array(MNIST_WIDTH_PX * MNIST_WIDTH_PX);
    for (let i = 0; i < data.length; i += 4) {
      const pixel = data[i];
      tensorData[i / 4] = 1 - pixel / 255;
    }

    // Draw the ascii art out to the screen
    let string = '';
    for (let i = 0; i < MNIST_WIDTH_PX; i++) {
      for (let j = 0; j < MNIST_WIDTH_PX; j++) {
        const index = i * MNIST_WIDTH_PX + j;
        if (tensorData[index] > 0.5) {
          string += 'X';
        } else {
          string += '.';
        }
      }
      string += '\n';
    }
    console.log(string);

    const result = model.predict(
      tf.tensor(tensorData).reshape([-1, 28, 28, 1]),
    ) as tf.Tensor;

    result.array().then((results: any) => {
      setModelResults(
        (results as [number[]])[0]
          .map((rank, number) => ({
            rank,
            number,
            generation: generation++,
          }))
          .filter((result) => result.rank > 0.05)
          .sort((a, b) => b.rank - a.rank),
      );
    });
  }, [curve]);

  function changeModel() {
    setModelPath(ensureExists(selectRef.current).value);
  }

  const nf = new Intl.NumberFormat('en-US');
  return (
    <div className="mnist">
      <h1>Draw a number</h1>
      <p>
        This project is demonstrating the results of trained tensorflow models
        for recognizing hand written digits. The model was trained on the mnist
        data set.
      </p>
      <select onChange={changeModel} ref={selectRef}>
        <option value="models/mnist-cnn/model.json">
          Convolutional Neural Network
        </option>
        <option value="models/mnist-model/model.json">
          Basic Feed Forward
        </option>
      </select>
      <canvas
        className="mnistCanvas"
        width={MNIST_WIDTH_PX}
        height={MNIST_WIDTH_PX}
        ref={canvasRef}
      />
      <div className="mnistResults">
        {modelResults?.map(({ rank, number, generation }) => (
          <div key={generation}>
            <span className="mnistResultsNumber">{number}</span>
            <span className="mnistResultsRank">{nf.format(rank)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

type Vec2<T = number> = { x: T; y: T };

type Curve = {
  line: Vec2[];
  cpLeft: Vec2[];
  cpRight: Vec2[];
  smoothness: number;
  distance: number;
};

type Config = {
  onPoint: (prev: Vec2, curr: Vec2) => void;
  onStartDraw: () => void;
  drawingTarget: HTMLElement;
  onCurveDrawn: (curve: Curve) => void;
  pointsPerDistance: number;
};

type Current = {
  points: Vec2[];
  distancePerPoint: number[];
  isDrawingCurve: boolean;
  totalLineDistance: number;
};

function setupCurveDrawing(config: Config) {
  const current: Current = {
    points: [],
    distancePerPoint: [],
    isDrawingCurve: false,
    totalLineDistance: 0,
  };

  const { drawingTarget } = config;

  drawingTarget.addEventListener('mousedown', onMouseDown);
  drawingTarget.addEventListener('touchmove', onTouchMove);
  drawingTarget.addEventListener('touchstart', onTouchStart);
  drawingTarget.style.touchAction = 'none';

  function getCoord(event: { clientX: number; clientY: number }) {
    const { clientX, clientY } = event;
    const { left, top } = drawingTarget.getBoundingClientRect();
    return [clientX - left, clientY - top];
  }

  function onTouchStart(event: TouchEvent): void {
    const target = event.target as null | HTMLElement;
    if (target && target.tagName === 'A') {
      // Don't start drawing on a link.
      return;
    }

    event.preventDefault();

    if (current.isDrawingCurve === false) {
      config.onStartDraw();
      drawingTarget.addEventListener('touchend', onTouchEnd);

      // Reset the current state
      current.isDrawingCurve = true;
      current.points = [];
      current.distancePerPoint = [];
      current.totalLineDistance = 0;

      const [x, y] = getCoord(event.touches[0]);
      addPoint(config, current, x, y);
    }
  }

  function onMouseDown(event: MouseEvent): void {
    const target = event.target as null | HTMLElement;
    if (target && target.tagName === 'A') {
      // Don't start drawing on a link.
      return;
    }

    if (current.isDrawingCurve === false) {
      config.onStartDraw();
      drawingTarget.addEventListener('mousemove', onMouseMove);
      drawingTarget.addEventListener('mouseout', onMouseMoveDone);
      drawingTarget.addEventListener('mouseup', onMouseMoveDone);

      current.isDrawingCurve = true;
      current.points = [];
      current.distancePerPoint = [];
      current.totalLineDistance = 0;
      const [x, y] = getCoord(event);
      addPoint(config, current, x, y);
    }
  }

  function onMouseMove(event: MouseEvent): void {
    event.preventDefault();
    const [x, y] = getCoord(event);
    addPoint(config, current, x, y);
  }

  function onTouchMove(event: TouchEvent): void {
    event.preventDefault();
    const [x, y] = getCoord(event.touches[0]);
    addPoint(config, current, x, y);
  }

  function onTouchEnd(): void {
    drawingTarget.removeEventListener('touchend', onTouchEnd);

    completeCurve(config, current);
  }

  function onMouseMoveDone(event: MouseEvent): void {
    drawingTarget.removeEventListener('mousemove', onMouseMove);
    drawingTarget.removeEventListener('mouseout', onMouseMoveDone);
    drawingTarget.removeEventListener('mouseup', onMouseMoveDone);

    const [x, y] = getCoord(event);
    addPoint(config, current, x, y);

    completeCurve(config, current);
  }

  return current;
}

function completeCurve(config: Config, current: Current): void {
  current.isDrawingCurve = false;

  const line = smoothLine(config, current);
  const curve = generateSmoothedBezierCurve(line, 0.3);
  config.onCurveDrawn(curve);
  current.points = [];
}

/**
 * Add point to to the current line.
 */
function addPoint(
  config: Config,
  current: Current,
  x: number,
  y: number,
): void {
  const curr = { x, y };

  let prev;
  if (current.points.length > 0) {
    prev = current.points[current.points.length - 1];
  } else {
    prev = curr;
  }

  const distance = Math.sqrt(
    Math.pow(prev.x - curr.x, 2) + Math.pow(prev.y - curr.y, 2),
  );

  current.totalLineDistance += distance;
  current.points.push(curr);
  current.distancePerPoint.push(distance);

  config.onPoint(prev, curr);
}

function smoothLine(config: Config, current: Current): Vec2[] {
  const { pointsPerDistance } = config;
  const { totalLineDistance, points, distancePerPoint } = current;
  const smoothPoints = [];
  let positionOnLinePiece = 0;
  let positionPrev = 0;
  let positionOnLine = 0;

  if (points.length <= 2) {
    return points;
  }

  let divisions = Math.ceil(totalLineDistance / pointsPerDistance);
  divisions = Math.max(2, divisions);
  const targetDistance = totalLineDistance / divisions;

  let i = 0;

  smoothPoints.push(points[0]); //Add the first point

  for (let j = 1; j < divisions; j++) {
    const distanceAtSegment = j * targetDistance;

    while (positionOnLine < distanceAtSegment) {
      i++;
      positionPrev = positionOnLine;
      positionOnLine += distancePerPoint[i];
    }

    positionOnLinePiece = positionOnLine - positionPrev;

    const theta = Math.atan2(
      points[i].y - points[i - 1].y,
      points[i].x - points[i - 1].x,
    );

    smoothPoints.push({
      x: points[i - 1].x + positionOnLinePiece * Math.cos(theta),
      y: points[i - 1].y + positionOnLinePiece * Math.sin(theta),
    });
  }

  smoothPoints.push(points[points.length - 1]); // Add the last point

  return smoothPoints;
}

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
function generateSmoothedBezierCurve(line: Vec2[], smoothness: number) {
  let distance = 0;
  const distances = [];
  const cpLeft: Vec2[] = [];
  const cpRight: Vec2[] = [];

  // Generate distances
  for (let i = 1; i < line.length; i++) {
    const segmentDistance = Math.sqrt(
      Math.pow(line[i - 1].x - line[i].x, 2) +
        Math.pow(line[i - 1].y - line[i].y, 2),
    );
    distances.push(segmentDistance);
    distance += distance;
  }

  // Add a beginning control point.
  const firstPoint = line[0];
  cpLeft.push({ ...firstPoint });
  cpRight.push({ ...firstPoint });

  // Generate control points.
  for (let i = 1; i < line.length - 1; i++) {
    const p1 = line[i - 1];
    const p2 = line[i];
    const p3 = line[i + 1];

    const d1 = distances[i - 1];
    const d2 = distances[i];

    const theta = Math.atan2(p3.y - p1.y, p3.x - p1.x);

    cpLeft.push({
      x: p2.x + d1 * smoothness * Math.cos(theta + Math.PI),
      y: p2.y + d1 * smoothness * Math.sin(theta + Math.PI),
    });

    cpRight.push({
      x: p2.x + d2 * smoothness * Math.cos(theta),
      y: p2.y + d2 * smoothness * Math.sin(theta),
    });
  }

  // Add an ending control point
  const lastPoint = line[line.length - 1];
  cpLeft.push({ ...lastPoint });
  cpRight.push({ ...lastPoint });

  return {
    line,
    cpLeft,
    cpRight,
    smoothness,
    distance,
  };
}

function drawCurve(ctx: CanvasRenderingContext2D, curve: Curve): void {
  ctx.lineWidth = LINE_WIDTH;
  ctx.strokeStyle = '#000';
  ctx.beginPath();
  ctx.lineCap = 'round';

  const { line, cpRight, cpLeft } = curve;
  const firstPoint = line[0];
  ctx.moveTo(firstPoint.x, firstPoint.y);

  for (let i = 1; i < line.length; i++) {
    ctx.bezierCurveTo(
      cpRight[i - 1].x,
      cpRight[i - 1].y,
      cpLeft[i].x,
      cpLeft[i].y,
      line[i].x,
      line[i].y,
    );
  }

  ctx.stroke();
}

function hslToFillStyle(h: number, s: number, l: number, a: number): string {
  if (a === undefined) {
    return ['hsl(', h, ',', s, '%,', l, '%)'].join('');
  }
  return ['hsla(', h, ',', s, '%,', l, '%,', a, ')'].join('');
}
