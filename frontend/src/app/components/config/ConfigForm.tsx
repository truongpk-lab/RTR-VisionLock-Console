import { useEffect, useMemo, useState } from "react";
import type { ComponentType } from "react";
import { Check, RotateCcw, Save } from "lucide-react";
import { useTrackingSession } from "../../lib/trackingSession";

// One tunable backend config value, addressed by its path from the config root
// (e.g. ["thresholds", "stable_threshold"] or ["samurai","memory_admission","min_affinity"]).
export interface FieldSpec {
  path: string[];
  label: string;
  hint?: string;
  kind?: "number" | "bool";
  min?: number;
  max?: number;
  step?: number;
}

export interface ConfigSection {
  heading?: string;
  fields: FieldSpec[];
}

function getByPath(obj: any, path: string[]): any {
  return path.reduce((acc, key) => (acc == null ? undefined : acc[key]), obj);
}

// Build a nested patch object ({thresholds: {stable_threshold: 0.7}}) from a flat draft.
function buildPatch(specs: FieldSpec[], draft: Record<string, any>): Record<string, any> {
  const patch: Record<string, any> = {};
  for (const spec of specs) {
    const key = spec.path.join(".");
    if (!(key in draft)) continue;
    let node = patch;
    for (let i = 0; i < spec.path.length - 1; i++) {
      const k = spec.path[i];
      node[k] = node[k] ?? {};
      node = node[k];
    }
    node[spec.path[spec.path.length - 1]] = draft[key];
  }
  return patch;
}

function decimals(step?: number): number {
  if (!step || step >= 1) return 0;
  return Math.max(0, Math.ceil(-Math.log10(step)));
}

export function ConfigForm({
  title,
  description,
  icon: Icon,
  sections,
  presets,
}: {
  title: string;
  description: string;
  icon?: ComponentType<{ size?: number; className?: string }>;
  sections: ConfigSection[];
  presets?: { label: string; patch: Record<string, any> }[];
}) {
  const { connected, getConfig, patchConfig, saveConfig } = useTrackingSession();
  const specs = useMemo(() => sections.flatMap((s) => s.fields), [sections]);
  const [draft, setDraft] = useState<Record<string, any>>({});
  const [loaded, setLoaded] = useState(false);
  const [status, setStatus] = useState<string>("");

  const load = useMemo(
    () => async () => {
      try {
        const cfg = await getConfig();
        const next: Record<string, any> = {};
        for (const spec of specs) {
          const value = getByPath(cfg, spec.path);
          if (value !== undefined) next[spec.path.join(".")] = value;
        }
        setDraft(next);
        setLoaded(true);
      } catch {
        setStatus("Không tải được config (backend offline?)");
      }
    },
    [getConfig, specs],
  );

  useEffect(() => {
    load();
  }, [load]);

  const setField = (key: string, value: any) => setDraft((d) => ({ ...d, [key]: value }));

  const applyPreset = async (preset: { label: string; patch: Record<string, any> }) => {
    try {
      await patchConfig(preset.patch);
      await load();
      setStatus(`Đã áp preset "${preset.label}".`);
    } catch {
      setStatus("Áp preset thất bại.");
    }
  };

  const apply = async (persist: boolean) => {
    try {
      await patchConfig(buildPatch(specs, draft));
      if (persist) {
        await saveConfig();
        setStatus("Đã áp dụng và lưu vào local.yaml.");
      } else {
        setStatus("Đã áp dụng (tạm thời, chưa lưu).");
      }
    } catch {
      setStatus("Áp dụng thất bại.");
    }
  };

  return (
    <div className="flex-1 h-full overflow-y-auto bg-[#05090f] text-slate-200">
      <div className="max-w-3xl mx-auto p-6">
        <div className="flex items-center gap-3 mb-1">
          {Icon && (
            <div className="w-9 h-9 rounded bg-cyan-950 border border-cyan-800/50 flex items-center justify-center text-cyan-400">
              <Icon size={18} />
            </div>
          )}
          <h1 className="text-lg font-mono font-bold tracking-wider text-slate-100">{title}</h1>
        </div>
        <p className="text-xs text-slate-500 mb-6">{description}</p>

        {!connected && (
          <div className="mb-4 px-3 py-2 text-[11px] font-mono text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded">
            Backend offline — bật camera/backend rồi tải lại trang.
          </div>
        )}

        {presets && presets.length > 0 && (
          <div className="mb-6 flex flex-wrap gap-2">
            {presets.map((preset) => (
              <button
                key={preset.label}
                onClick={() => applyPreset(preset)}
                disabled={!loaded}
                className="px-3 py-1.5 text-xs font-mono uppercase tracking-wider bg-slate-800 hover:bg-cyan-900/50 border border-slate-700 hover:border-cyan-700/60 text-slate-300 hover:text-cyan-300 rounded disabled:opacity-40 transition-colors"
              >
                {preset.label}
              </button>
            ))}
          </div>
        )}

        {sections.map((section, si) => (
          <div key={si} className="mb-6">
            {section.heading && (
              <div className="text-[10px] font-mono uppercase tracking-widest text-cyan-500/70 mb-3 border-b border-slate-800/60 pb-1">
                {section.heading}
              </div>
            )}
            <div className="flex flex-col gap-4">
              {section.fields.map((spec) => {
                const key = spec.path.join(".");
                const value = draft[key];
                if (spec.kind === "bool") {
                  return (
                    <div key={key} className="flex items-center justify-between gap-4">
                      <div>
                        <div className="text-sm text-slate-200">{spec.label}</div>
                        {spec.hint && <div className="text-[11px] text-slate-500">{spec.hint}</div>}
                      </div>
                      <button
                        type="button"
                        disabled={!loaded}
                        onClick={() => setField(key, !value)}
                        className={`w-11 h-6 rounded-full transition-colors flex-shrink-0 relative ${
                          value ? "bg-cyan-500/80" : "bg-slate-700"
                        }`}
                      >
                        <span
                          className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-all ${
                            value ? "left-[22px]" : "left-0.5"
                          }`}
                        />
                      </button>
                    </div>
                  );
                }
                const dp = decimals(spec.step);
                return (
                  <div key={key}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm text-slate-200">{spec.label}</span>
                      <span className="font-mono text-cyan-300 text-sm tabular-nums">
                        {typeof value === "number" ? value.toFixed(dp) : "—"}
                      </span>
                    </div>
                    <input
                      type="range"
                      disabled={!loaded || typeof value !== "number"}
                      min={spec.min ?? 0}
                      max={spec.max ?? 1}
                      step={spec.step ?? 0.01}
                      value={typeof value === "number" ? value : 0}
                      onChange={(e) => setField(key, Number(e.target.value))}
                      className="w-full accent-cyan-500 cursor-pointer"
                    />
                    {spec.hint && <div className="text-[11px] text-slate-500 mt-0.5">{spec.hint}</div>}
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        <div className="sticky bottom-0 -mx-6 px-6 py-3 bg-[#05090f]/95 border-t border-slate-800/60 flex items-center gap-3 backdrop-blur-sm">
          <button
            onClick={() => apply(false)}
            disabled={!loaded}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 text-white text-sm font-medium rounded transition-colors"
          >
            <Check size={15} /> Apply
          </button>
          <button
            onClick={() => apply(true)}
            disabled={!loaded}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 text-sm font-medium rounded transition-colors"
          >
            <Save size={15} /> Apply + Save
          </button>
          <button
            onClick={load}
            disabled={!loaded}
            className="flex items-center gap-2 px-3 py-2 text-slate-400 hover:text-slate-200 text-sm rounded transition-colors"
          >
            <RotateCcw size={15} /> Revert
          </button>
          {status && <span className="ml-auto text-[11px] font-mono text-slate-400">{status}</span>}
        </div>
      </div>
    </div>
  );
}
