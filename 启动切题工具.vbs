Option Explicit

Dim shell, fso, processEnv, folder, workbenchFolder, workbench, item
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
Set processEnv = shell.Environment("PROCESS")

' Electron must not inherit Node-only mode from terminals or developer tools.
On Error Resume Next
processEnv.Remove("ELECTRON_RUN_AS_NODE")
On Error GoTo 0

folder = fso.GetParentFolderName(WScript.ScriptFullName)
workbenchFolder = folder & "\desktop_annotation\PaperFlow-Annotation"
workbench = ""

' Locate the product executable by its ASCII prefix.  This avoids hard-coding
' a UTF-8 Chinese filename while never falling back to the legacy OCR binary.
If fso.FolderExists(workbenchFolder) Then
    For Each item In fso.GetFolder(workbenchFolder).Files
        If LCase(fso.GetExtensionName(item.Name)) = "exe" _
           And LCase(Left(item.Name, 9)) = "paperflow" Then
            workbench = item.Path
            Exit For
        End If
    Next
End If

If workbench <> "" Then
    shell.CurrentDirectory = workbenchFolder
    shell.Run """" & workbench & """", 1, False
    ' On a slow first Electron launch the native window can open behind Explorer.
    ' Bring the first window forward so a second double-click is unnecessary.
    WScript.Sleep 700
    shell.AppActivate "PaperFlow"
Else
    MsgBox "PaperFlow workbench is missing. Please contact the administrator.", 16, "PaperFlow"
End If
