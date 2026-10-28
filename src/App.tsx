import { HashRouter, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/AppLayout";
import { Dashboard } from "./views/Dashboard";
import { Projects } from "./views/Projects";
import { Scene } from "./views/Scene";
import { Settings } from "./views/Settings";
import "./App.css";

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="projects" element={<Projects />} />
          <Route path="scene" element={<Scene />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Dashboard />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}

export default App;
