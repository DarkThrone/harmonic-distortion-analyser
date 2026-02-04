import { useState, useMemo, useRef, useEffect } from "react";

const { sign, abs, tanh, atan, sin, min, max, PI, cos, sqrt, log10 } = Math;

type ClipFn = (x: number, t?: number) => number;

type Clipper = {
  name: string;
  fn: ClipFn;
  desc: string;
};
function knee(clipFn: ClipFn) {
  return (x: number, t = 0) => {
    const st = sign(x) * t;
    const k = 1 - t;

    return abs(x) < t ? x : st + k * clipFn((x - st) / k);
  };
}

function quad(x: number) {
  return x < -1 ? -2 / 3 : x > 1 ? 2 / 3 : x - (x * x * x) / 3;
}

function softAtan(x: number) {
  return (2 / PI) * atan((PI / 2) * x);
}
const baseline: Clipper = {
  name: "Clean",
  fn: (x) => x,
  desc: "Pure sine — single harmonic",
};

// Clipping algorithms
const clippers: Record<string, Clipper> = {
  none: {
    name: "Clean",
    fn: knee((x) => x),
    desc: "Pure sine — single harmonic",
  },
  hard: {
    name: "Hard Clip",
    fn: knee((x) => max(-1, min(1, x))),
    desc: "Abrupt cutoff — strong odd harmonics",
  },
  softTanh: {
    name: "Soft (tanh)",
    fn: knee(tanh),
    desc: "Smooth saturation — odd harmonics, gentler rolloff",
  },
  softCubic: {
    name: "Soft (cubic)",
    fn: knee(quad),
    desc: "Polynomial — primarily 3rd harmonic",
  },
  softArctan: {
    name: "Soft (arctan)",
    fn: knee(softAtan),
    desc: "Smoother saturation with arctan",
  },
};

// Generate pure sine wave with clipping applied
function generateWaveform(
  clipFn: ClipFn,
  samples = 1024,
  frequency = 4,
  drive = 1,
  knee = 0,
): Array<number> {
  const output = new Array(samples);

  for (let i = 0; i < samples; i++) {
    const t = i / samples;
    const sine = sin(2 * PI * frequency * t);
    output[i] = clipFn(sine * drive, knee);
  }

  return output;
}

function computeResidual(
  waveform: Array<number>,
  baseline: Array<number>,
): Array<number> {
  const result = new Array<number>();

  if (waveform.length !== baseline.length) {
    throw new Error("Sample mismatch.");
  }

  for (let i = 0, l = waveform.length; i < l; i++) {
    result.push(baseline[i] - waveform[i]);
  }

  return result;
}

// DFT for spectrum analysis
function computeSpectrum(
  waveform: Array<number>,
  numHarmonics = 16,
): Array<{
  harmonic: number;
  magnitude: number;
  db: number;
}> {
  const n = waveform.length;
  const fundamentalBin = 6;
  const spectrum = [];

  for (let h = 0; h <= numHarmonics; h++) {
    const k = fundamentalBin * h;
    if (k >= n / 2) break;

    let real = 0,
      imag = 0;
    for (let t = 0; t < n; t++) {
      const angle = (2 * PI * k * t) / n;
      real += waveform[t] * cos(angle);
      imag -= waveform[t] * sin(angle);
    }
    const magnitude = (sqrt(real * real + imag * imag) * 2) / n;
    spectrum.push({
      harmonic: h,
      magnitude: magnitude,
      db: magnitude > 0.0001 ? 20 * log10(magnitude) : 0,
    });
  }

  return spectrum;
}

function computeRootMeanSquare(sample: Array<number>): number {
  const inverseTotal = 1 / sample.length;
  return sqrt(
    inverseTotal * sample.map((s) => s * s).reduce((a, s) => a + s, 0),
  );
}

function App() {
  // Canvas waveform renderer
  function WaveformCanvas({
    data,
    referenceData,
    color,
    height = 150,
  }: {
    data: Array<number>;
    referenceData?: Array<number>;
    color: string;
    height: number;
  }) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(
      null,
    );

    useEffect(() => {
      const canvas = canvasRef.current!;

      const handleMouseMove = (e: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        setMousePos({ x, y });
      };

      const handleMouseLeave = () => {
        setMousePos(null);
      };

      canvas.addEventListener("mousemove", handleMouseMove);
      canvas.addEventListener("mouseleave", handleMouseLeave);

      return () => {
        canvas.removeEventListener("mousemove", handleMouseMove);
        canvas.removeEventListener("mouseleave", handleMouseLeave);
      };
    }, []);

    useEffect(() => {
      const canvas = canvasRef.current!;

      const ctx = canvas.getContext("2d")!;
      const width = canvas.width;

      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      for (let y = 0; y <= 4; y++) {
        ctx.beginPath();
        ctx.moveTo(0, (y * height) / 4);
        ctx.lineTo(width, (y * height) / 4);
        ctx.stroke();
      }

      // Scale factor for [-6, 6] range (instead of [-1, 1])
      const scale = (height * 0.4) / 6;

      // Draw reference waveform if provided
      if (referenceData) {
        const step = width / referenceData.length;
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        for (let i = 0; i < referenceData.length; i++) {
          const x = i * step;
          const y = height / 2 - referenceData[i] * scale;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Show all 6 cycles
      const displayData = data;

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();

      const step = width / displayData.length;
      for (let i = 0; i < displayData.length; i++) {
        const x = i * step;
        const y = height / 2 - displayData[i] * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Add axis labels
      ctx.fillStyle = "#64748b";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("Amplitude", 8, 20);

      ctx.textAlign = "right";
      ctx.font = "12px sans-serif";
      ctx.fillText("+6", width - 8, height / 2 - height * 0.4 + 5);
      ctx.fillText("0", width - 8, height / 2 + 5);
      ctx.fillText("-6", width - 8, height / 2 + height * 0.4 + 5);

      ctx.font = "14px sans-serif";

      ctx.textAlign = "center";
      ctx.fillText("Time →", width / 2, height - 8);

      // Draw crosshair and tooltip for hovered point
      if (mousePos) {
        const step = width / data.length;

        // Find closest sample index
        const sampleIndex = Math.round(mousePos.x / step);

        if (sampleIndex >= 0 && sampleIndex < data.length) {
          const amplitude = data[sampleIndex];
          const x = sampleIndex * step;
          const y = height / 2 - amplitude * scale;

          // Calculate time in cycles (6 cycles total)
          const timeInCycles = (sampleIndex / data.length) * 6;

          // Draw vertical crosshair line
          ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
          ctx.setLineDash([]);

          // Draw point on waveform
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * PI);
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();

          // Draw tooltip
          const tooltipText = `Time: ${timeInCycles.toFixed(3)} cycles`;
          const tooltipAmp = `Amplitude: ${amplitude.toFixed(3)}`;

          ctx.font = "14px sans-serif";
          const textWidth = max(
            ctx.measureText(tooltipText).width,
            ctx.measureText(tooltipAmp).width,
          );
          const tooltipWidth = textWidth + 20;
          const tooltipHeight = 50;

          // Position tooltip near mouse, but keep it in bounds
          let tooltipX = mousePos.x + 15;
          let tooltipY = mousePos.y - 10;

          if (tooltipX + tooltipWidth > width - 10) {
            tooltipX = mousePos.x - tooltipWidth - 15;
          }
          if (tooltipY < 10) {
            tooltipY = 10;
          }
          if (tooltipY + tooltipHeight > height - 10) {
            tooltipY = height - tooltipHeight - 10;
          }

          // Draw tooltip background
          ctx.fillStyle = "rgba(15, 23, 42, 0.95)";
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.fillRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);
          ctx.strokeRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);

          // Draw tooltip text
          ctx.fillStyle = "#ffffff";
          ctx.textAlign = "left";
          ctx.fillText(tooltipText, tooltipX + 10, tooltipY + 22);
          ctx.fillStyle = "#94a3b8";
          ctx.font = "12px sans-serif";
          ctx.fillText(tooltipAmp, tooltipX + 10, tooltipY + 38);
        }
      }
    }, [data, referenceData, color, height, mousePos]);

    return (
      <canvas
        ref={canvasRef}
        width={1000}
        height={height}
        className="rounded-lg w-full cursor-crosshair"
      />
    );
  }

  // Transfer function visualizer
  function TransferCurve({
    clipFn,
    drive,
    color,
  }: {
    clipFn: ClipFn;
    drive: number;
    color: string;
  }) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      const size = 250;

      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, size, size);

      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, size / 2);
      ctx.lineTo(size, size / 2);
      ctx.moveTo(size / 2, 0);
      ctx.lineTo(size / 2, size);
      ctx.stroke();

      // Linear reference line
      ctx.strokeStyle = "#334155";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, size);
      ctx.lineTo(size, 0);
      ctx.stroke();

      // Transfer curve
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let x = 0; x < size; x++) {
        const input = (x / size) * 4 - 2;
        const output = clipFn(input);
        const y = size / 2 - (output * size) / 4;
        if (x === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Add axis labels
      ctx.fillStyle = "#64748b";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("Output", 8, 20);

      ctx.textAlign = "center";
      ctx.fillText("Input →", size / 2, size - 8);

      // Add scale markers
      ctx.font = "12px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("-2", 5, size - 5);

      ctx.textAlign = "right";
      ctx.fillText("+2", size - 5, 15);
      ctx.fillText("+2", size - 5, size / 2 - 5);
      ctx.fillText("-2", size - 5, size - 5);
    }, [clipFn, drive, color]);

    return (
      <canvas ref={canvasRef} width={250} height={250} className="rounded-lg" />
    );
  }

  // Harmonic bar chart using canvas
  function HarmonicBars({
    spectrum,
    color,
    height = 120,
  }: {
    spectrum: Array<{ magnitude: number; db: number; harmonic: number }>;
    color: string;
    height: number;
  }) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const [hoveredBar, setHoveredBar] = useState<number | null>(null);
    const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(
      null,
    );

    useEffect(() => {
      const canvas = canvasRef.current!;

      const handleMouseMove = (e: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        setMousePos({ x, y });
      };

      const handleMouseLeave = () => {
        setHoveredBar(null);
        setMousePos(null);
      };

      canvas.addEventListener("mousemove", handleMouseMove);
      canvas.addEventListener("mouseleave", handleMouseLeave);

      return () => {
        canvas.removeEventListener("mousemove", handleMouseMove);
        canvas.removeEventListener("mouseleave", handleMouseLeave);
      };
    }, []);

    useEffect(() => {
      const canvas = canvasRef.current!;

      const ctx = canvas.getContext("2d")!;
      const width = canvas.width;
      const h = canvas.height;

      // Clear
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, width, h);

      // Define padding
      const paddingTop = 35;
      const paddingBottom = 50;
      const paddingLeft = 20;
      const paddingRight = 50;

      // Define dBFS scale
      const maxDb = 30;
      const minDb = -80;

      // Calculate chart area
      const chartHeight = h - paddingTop - paddingBottom;

      // Position 0dB axis at ~25% from top (30dB range above, 80dB range below)
      const axisY = paddingTop + chartHeight * 0.25;
      const topSectionHeight = chartHeight * 0.25; // 0 to +30 dB
      const bottomSectionHeight = chartHeight * 0.75; // -80 to 0 dB

      // Draw 0 dB axis line
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, axisY);
      ctx.lineTo(width - paddingRight, axisY);
      ctx.stroke();

      // Draw bars for harmonics 0-15 (including DC component)
      const harmonics = spectrum.slice(0, 16);
      const barWidth = (width - paddingLeft - paddingRight) / harmonics.length;

      // Detect which bar is being hovered
      let detectedHover: number | null = null;
      if (mousePos) {
        harmonics.forEach((_, i) => {
          const x = paddingLeft + i * barWidth;
          if (mousePos.x >= x && mousePos.x <= x + barWidth) {
            detectedHover = i;
          }
        });
      }
      if (detectedHover !== hoveredBar) {
        setHoveredBar(detectedHover);
      }

      harmonics.forEach((s, i) => {
        const x = paddingLeft + i * barWidth;
        const db = s?.db;

        // Skip if db is invalid (null, undefined, NaN, or Infinity)
        if (db == null || !isFinite(db)) {
          return;
        }

        let y, barHeight;

        if (db >= 0) {
          // Positive dB: scale from 0 to +30 dB
          const normalizedDb = min(db, maxDb);
          barHeight = (normalizedDb / maxDb) * topSectionHeight;
          y = axisY - barHeight;
        } else {
          // Negative dB: scale from 0 to -80 dB
          const normalizedDb = max(db, minDb);
          barHeight = (abs(normalizedDb) / abs(minDb)) * bottomSectionHeight;
          y = axisY;
        }

        // Ensure barHeight is valid before drawing
        if (!isFinite(barHeight) || barHeight < 0) {
          return;
        }

        // Determine if this bar is hovered
        const isHovered = i === hoveredBar;

        // Bar
        ctx.fillStyle = abs(db) > -70 ? color : "#334155";
        ctx.globalAlpha = isHovered ? 1 : abs(db) > -70 ? 0.8 : 0.3;
        ctx.fillRect(x + 4, y, barWidth - 8, abs(barHeight));

        // Highlight hovered bar with border
        if (isHovered) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 4, y, barWidth - 8, abs(barHeight));
        }

        ctx.globalAlpha = 1;

        // Harmonic number label
        const harmonicNum = s.harmonic;
        const isOdd = harmonicNum % 2 === 1;
        ctx.fillStyle = isOdd ? "#94a3b8" : "#64748b";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(
          harmonicNum.toString(),
          x + barWidth / 2,
          h - paddingBottom + 20,
        );
      });

      // Add axis labels
      ctx.fillStyle = "#64748b";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("dBFS", paddingLeft, 18);

      ctx.textAlign = "center";
      ctx.fillText("Harmonic Number →", width / 2, h - 10);

      // Add dB scale
      ctx.textAlign = "right";
      ctx.font = "12px sans-serif";
      ctx.fillText("0 dB", width - 5, axisY + 5);
      ctx.fillText(`+${maxDb}`, width - 5, paddingTop + 5);
      ctx.fillText(`${minDb}`, width - 5, paddingTop + chartHeight + 5);

      // Draw tooltip for hovered bar
      if (hoveredBar !== null && mousePos) {
        const s = harmonics[hoveredBar];
        if (s && s.db != null && isFinite(s.db)) {
          const tooltipText = `H${s.harmonic}: ${s.db.toFixed(2)} dBFS`;
          const tooltipMag = `Mag: ${s.magnitude.toFixed(4)}`;

          // Measure text for tooltip background
          ctx.font = "14px sans-serif";
          const textWidth = max(
            ctx.measureText(tooltipText).width,
            ctx.measureText(tooltipMag).width,
          );
          const tooltipWidth = textWidth + 20;
          const tooltipHeight = 50;

          // Position tooltip near mouse, but keep it in bounds
          let tooltipX = mousePos.x + 15;
          let tooltipY = mousePos.y - 10;

          if (tooltipX + tooltipWidth > width - 10) {
            tooltipX = mousePos.x - tooltipWidth - 15;
          }
          if (tooltipY < 10) {
            tooltipY = 10;
          }
          if (tooltipY + tooltipHeight > h - 10) {
            tooltipY = h - tooltipHeight - 10;
          }

          // Draw tooltip background
          ctx.fillStyle = "rgba(15, 23, 42, 0.95)";
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.fillRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);
          ctx.strokeRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);

          // Draw tooltip text
          ctx.fillStyle = "#ffffff";
          ctx.textAlign = "left";
          ctx.fillText(tooltipText, tooltipX + 10, tooltipY + 22);
          ctx.fillStyle = "#94a3b8";
          ctx.font = "12px sans-serif";
          ctx.fillText(tooltipMag, tooltipX + 10, tooltipY + 38);
        }
      }
    }, [spectrum, color, height, hoveredBar, mousePos]);

    return (
      <canvas
        ref={canvasRef}
        width={1000}
        height={height}
        className="rounded-lg w-full cursor-pointer"
      />
    );
  }

  // Harmonic table
  function HarmonicTable({
    spectrum,
    color,
  }: {
    spectrum: Array<{ db: number; harmonic: number }>;
    color: string;
  }) {
    return (
      <div className="text-xs font-mono">
        <div className="grid grid-cols-8 gap-1">
          {spectrum.slice(0, 8).map((s, i) => (
            <div key={i} className="text-center">
              <div className="text-slate-500">H{s.harmonic}</div>
              <div style={{ color: s.db > -80 ? color : "#475569" }}>
                {s.db > -80 ? s.db.toFixed(1) : "—"}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  function OscillatorSim() {
    const [selectedClippers, setSelectedClippers] = useState([
      "hard",
      "softTanh",
      "softCubic",
      "softArctan",
    ]);
    const [drive, setDrive] = useState(1);
    const [knee, setKnee] = useState(0);

    const colors = [
      "#3b82f6",
      "#ec4899",
      "#10b981",
      "#f59e0b",
      "#8b5cf6",
      "#ef4444",
      "#06b6d4",
      "#84cc16",
      "#f97316",
    ];

    const toggleClipper = (key: string) => {
      setSelectedClippers((prev) =>
        prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
      );
    };

    // Generate original unclipped sine wave
    const originalWaveform = useMemo(() => {
      return generateWaveform(clippers.none.fn, 1024, 6, drive, knee);
    }, [drive, knee]);
    const baselineWaveform = generateWaveform(baseline.fn, 1024, 6, drive);
    const waveforms = useMemo(() => {
      const results = selectedClippers.map((key) => {
        const waveform = generateWaveform(clippers[key].fn, 1024, 6, drive);
        const spectrum = computeSpectrum(waveform);
        const distortion = computeResidual(waveform, baselineWaveform);
        const residualRMS = computeRootMeanSquare(distortion);

        return {
          key,
          data: waveform,
          spectrum,
          distortion,
          residualRMS,
          clipper: clippers[key],
        };
      });
      return results;
    }, [selectedClippers, drive]);

    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-2xl font-bold mb-1">
            Sine Wave Distortion Harmonics
          </h1>
          <p className="text-slate-400 mb-6">
            Pure sine input → clipping → see which harmonics appear
          </p>

          <div className="mb-6 bg-slate-900 rounded-lg p-4 inline-block">
            <label className="block text-sm text-slate-400 mb-2">
              Input Drive:{" "}
              <span className="text-white font-mono">{drive.toFixed(2)}×</span>
              <span className="text-slate-500 ml-2">
                {drive < 1.1
                  ? "(minimal distortion)"
                  : drive < 2
                    ? "(moderate)"
                    : "(heavy distortion)"}
              </span>
            </label>
            <input
              type="range"
              min="0.5"
              max="6"
              step="0.05"
              value={drive}
              onChange={(e) => setDrive(parseFloat(e.target.value))}
              className="w-80 accent-blue-500"
            />
          </div>
          <div className="mb-6 bg-slate-900 rounded-lg p-4 inline-block">
            <label className="block text-sm text-slate-400 mb-2">
              Soft knee:{" "}
              <span className="text-white font-mono">{knee.toFixed(2)}×</span>
              <span className="text-slate-500 ml-2">
                {knee < 1.1
                  ? "(minimal distortion)"
                  : knee < 2
                    ? "(moderate)"
                    : "(heavy distortion)"}
              </span>
            </label>
            <input
              type="range"
              min="0.01"
              step="0.01"
              max="1"
              value={knee}
              onChange={(e) => setKnee(parseFloat(e.target.value))}
              className="w-80 accent-blue-500"
            />
          </div>
          <div className="mb-6">
            <p className="text-sm text-slate-400 mb-2">
              Select clipping types:
            </p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(clippers).map(([key, clip]) => (
                <button
                  key={key}
                  onClick={() => toggleClipper(key)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    selectedClippers.includes(key)
                      ? "bg-blue-600 text-white shadow-lg shadow-blue-500/25"
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                  }`}
                >
                  {clip.name}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            {waveforms.map(
              (
                { key, data, distortion, residualRMS, spectrum, clipper },
                i,
              ) => (
                <div
                  key={key}
                  className="bg-slate-900/50 rounded-xl p-4 border border-slate-800"
                >
                  <div className="flex items-center gap-3 mb-3">
                    <span
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: colors[i % colors.length] }}
                    />
                    <span className="font-semibold">{clipper.name}</span>
                    <span className="text-sm text-slate-500">
                      {clipper.desc}
                    </span>
                  </div>

                  <div className="grid grid-cols-[250px_1fr] gap-4">
                    <div>
                      <p className="text-xs text-slate-500 mb-1">
                        Transfer Function
                      </p>
                      <TransferCurve
                        clipFn={clipper.fn}
                        drive={drive}
                        color={colors[i % colors.length]}
                      />
                    </div>
                    <div>
                      <p className="text-xs text-slate-500 mb-1">
                        Waveform (6 cycles)
                      </p>
                      <WaveformCanvas
                        data={data}
                        referenceData={originalWaveform}
                        color={colors[i % colors.length]}
                        height={250}
                      />
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 mb-1">Distortion</p>
                    <WaveformCanvas
                      data={distortion}
                      referenceData={originalWaveform}
                      color={colors[i % colors.length]}
                      height={250}
                    />
                  </div>
                  <div className="mt-4">
                    <p className="text-xs text-slate-500 mb-1">
                      Harmonic Content
                    </p>
                    <HarmonicBars
                      spectrum={spectrum}
                      color={colors[i % colors.length]}
                      height={240}
                    />
                  </div>

                  <div className="mt-3 pt-3 border-t border-slate-800">
                    <HarmonicTable
                      spectrum={spectrum}
                      color={colors[i % colors.length]}
                    />
                  </div>
                </div>
              ),
            )}
          </div>
        </div>
      </div>
    );
  }

  return <OscillatorSim />;
}

export default App;
